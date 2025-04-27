import jax
import jax.numpy as jnp

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"

jax.config.update("jax_compilation_cache_dir", "cache/jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

import optax

from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n_embd", type=int, default=768)
parser.add_argument("--n_layer", type=int, default=12)
parser.add_argument("--vocab_size", type=int, default=8192)
parser.add_argument("--ctx_len", type=int, default=256)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--lr", type=float, default=2e-4)
args = parser.parse_args()

N, L, V, CTX_LEN, TRAIN_BATCH, LR = args.n_embd, args.n_layer, args.vocab_size, args.ctx_len, args.batch_size, args.lr

def xavier_uniform(key, shape, dtype):
    scale = jnp.sqrt(6/(shape[-1] + shape[-2]))
    return jax.random.uniform(key=key, shape=shape, minval=-scale, maxval=scale, dtype=dtype)

def rand_init(key, n_embd, n_layer, vocab_size, ctx_len, dtype=jnp.float32):
    _key1, _key2, _key3, _key4, _key5, _key6, _key7, _key8, _key9 = jax.random.split(key, 9)
    params = {}
    params['pos_embed'] = {'weight': xavier_uniform(_key1, (ctx_len, n_embd), dtype)}
    params['emb'] = {'weight': xavier_uniform(_key2, (vocab_size, n_embd), dtype)}
    params['head'] = {'weight': xavier_uniform(_key3, (vocab_size, n_embd), dtype), 'bias': jnp.zeros(vocab_size, dtype)}

    params['blocks'] = {
        'attn': {
            'query': {'weight': xavier_uniform(_key4, (n_layer, n_embd, n_embd), dtype)},
            'key': {'weight': xavier_uniform(_key5, (n_layer, n_embd, n_embd), dtype)},
            'value': {'weight': xavier_uniform(_key6, (n_layer, n_embd, n_embd), dtype)},
            'out': {'weight': xavier_uniform(_key7, (n_layer, n_embd, n_embd), dtype)}
        },
        'mlp': {
            'ff0': {'weight': xavier_uniform(_key8, (n_layer, 4 * n_embd, n_embd), dtype), 'bias': jnp.zeros((n_layer, 4 * n_embd), dtype)},
            'ff1': {'weight': xavier_uniform(_key9, (n_layer, n_embd, 4 * n_embd), dtype), 'bias': jnp.zeros((n_layer, n_embd), dtype)}
        },
        'ln1': {'bias': jnp.zeros((n_layer, n_embd), dtype), 'weight': jnp.ones((n_layer, n_embd), dtype)},
        'ln2': {'bias': jnp.zeros((n_layer, n_embd), dtype), 'weight': jnp.ones((n_layer, n_embd), dtype)}
    }
    return params


def layer_norm(x, w, eps=1e-5):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.var(x, axis=-1, keepdims=True)
    std = jnp.sqrt(var + eps)
    return (x - mean) / std * w['weight'] + w['bias']

def attention(x, params):
    T = x.shape[0]
    S = 64
    H = x.shape[1] // S
    q = jnp.reshape(x @ params['query']['weight'].T, (T, H, S))
    k = jnp.reshape(x @ params['key']['weight'].T, (T, H, S))
    v = jnp.reshape(x @ params['value']['weight'].T, (T, H, S))
    attn_output = jnp.reshape(jax.nn.dot_product_attention(q, k, v, is_causal=True), x.shape)
    return attn_output @ params['out']['weight'].T

def mlp_forward(x, params):
    x = x @ params['ff0']['weight'].T + params['ff0']['bias']
    x = jax.nn.gelu(x)
    x = x @ params['ff1']['weight'].T + params['ff1']['bias']
    return x

def forward(params, tokens):
    x = params['emb']['weight'][tokens] + params['pos_embed']['weight']
    @partial(jax.checkpoint, policy=jax.checkpoint_policies.dots_with_no_batch_dims_saveable)
    def block_loop(x, block):
        x = x + attention(layer_norm(x, block['ln1']), block['attn'])
        x = x + jax.vmap(mlp_forward, in_axes=(0, None))(layer_norm(x, block['ln2']), block['mlp'])
        return x, 0
    x, _ = jax.lax.scan(block_loop, x, params['blocks'])
    return x @ params['head']['weight'].T + params['head']['bias']

def loss(params, tokens):
    logits = forward(params, tokens)[:-1]
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, tokens[1:]))

def batch_loss(params, tokens):
    return jnp.mean(jax.vmap(loss, in_axes=(None, 0))(params, tokens))

fast_batch_grad = jax.value_and_grad(batch_loss)
