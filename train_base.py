import jax
import jax.numpy as jnp
from common import rand_init, fast_batch_grad, N, L, V, CTX_LEN, TRAIN_BATCH, LR

import optax

from tqdm import trange
import time

import numpy as np

params = rand_init(jax.random.key(0), N, L, V, CTX_LEN)
train_batch = jnp.reshape(jnp.arange(CTX_LEN * TRAIN_BATCH) % V, (TRAIN_BATCH, CTX_LEN))

print("Number of parameters:", jax.tree.reduce(lambda *x: sum(x), jax.tree.map(jnp.size, params)) / 1000000, "million")
solver = optax.adamw(LR)
# solver = optax.contrib.dadapt_adamw(1.0)
optimizer = solver.init(jax.tree.map(lambda p: p.copy(), params))

def do_update(params, optimizer, tokens_batch):
    loss, grad = fast_batch_grad(params, tokens_batch)
    updates, optimizer = solver.update(grad, optimizer, params)
    params = optax.apply_updates(params, updates)
    return params, optimizer, loss

print("Starting compilation")
start_time = time.time()
fast_update = jax.jit(do_update, donate_argnums=(0, 1)).lower(params, optimizer, train_batch).compile()
print("Compilation takes", time.time() - start_time)
print("Memory usage with", train_batch.shape)
print(fast_update.memory_analysis())

print(train_batch.shape)

for i in trange(100):
    params, optimizer, loss = fast_update(params, optimizer, train_batch)
    print(f"iter {i}: {loss}")
