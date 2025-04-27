import jax
import jax.numpy as jnp
from common import rand_init, fast_batch_grad, N, L, V, CTX_LEN, TRAIN_BATCH, LR

import optax

from tqdm import trange
import time

import numpy as np

from functools import partial, partialmethod

from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, SingleDeviceSharding
from jax.sharding import PartitionSpec as P

device_array = np.array(jax.devices())
mesh = Mesh(device_array, ("data",))

from dataclasses import dataclass, field

@jax.tree_util.register_dataclass
@dataclass
class Partitioned:
    v: jax.Array  # defaults to non-static data field
    idx: int = field(metadata=dict(static=True))  # marked as static meta field.

    def __getattr__(self, name):
        return getattr(gather_params(self), name) # pass through all attributes

def inner_fn(self, attr, *args, **kwargs):
    return getattr(gather_params(self), attr)(*args, **kwargs)

for attr in dir(jax.Array):
    if (attr not in dir(Partitioned) or attr in ['__lt__', '__le__', '__eq__', '__ne__', '__gt__', '__ge__']) and callable(getattr(jax.Array, attr)):
        setattr(Partitioned, attr, partialmethod(inner_fn, attr))
    
def shard_param(value, axis_name: str, min_weight_size: int = 2**18):
    if value.size < min_weight_size:
        return Partitioned(jax.device_put(value, NamedSharding(mesh, P())), 0)
    axis_size = mesh.shape[axis_name]
    shape = value.shape
    idx = np.argsort(shape)[::-1]  # Shard along largest possible axis.
    for i in idx:
        if shape[i] % axis_size == 0:
            return Partitioned(jax.device_put(value, NamedSharding(mesh, P(*([None] * i + [axis_name])))), i-len(shape))
    jax.debug.print("Skipping param due to incorrect size")
    return Partitioned(jax.device_put(value, NamedSharding(mesh, P())), 0)

def gather_array_with_mean_grads(x: jax.Array, axis: int, axis_name: str):
    axis_size = jax.lax.psum(1, axis_name)
    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            return (jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True) / axis_size)
        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn
    return f(x)

def gather_param(value, axis_name: str):
    if value.idx == 0:
        return value.v
    else:
        return gather_array_with_mean_grads(value.v, axis=len(value.v.shape)+value.idx, axis_name=axis_name)

def gather_params(params):
    return jax.tree.map(lambda x: gather_param(x, 'data'), params, is_leaf=lambda x: isinstance(x, Partitioned))

def sync_grads(value):
    if value.idx == 0:
        return Partitioned(jax.lax.pmean(value.v, 'data'), value.idx)
    else:
        return value

params = rand_init(jax.random.key(0), N, L, V, CTX_LEN)
params = jax.tree.map(lambda x: shard_param(x, "data"), params)

train_batch = jnp.reshape(jnp.arange(CTX_LEN * TRAIN_BATCH), (TRAIN_BATCH, CTX_LEN))
train_batch = jax.device_put(train_batch, NamedSharding(mesh, P('data', None)))

print("Number of parameters:", jax.tree.reduce(lambda *x: sum(x), jax.tree.map(jnp.size, params)) / 1000000, "million")
solver = optax.adamw(LR)
# solver = optax.contrib.dadapt_adamw(1.0) # WARNING: dadapt_adamw and other methods that do not strictly use elementwise accumulations will not be correct
optimizer = solver.init(jax.tree.map(lambda p: p.copy(), params))
optimizer = jax.tree.map(lambda x: jax.device_put(x, NamedSharding(mesh, P())) if isinstance(x.sharding, SingleDeviceSharding) else x, optimizer)

param_sharding = jax.tree.map(lambda x: x.sharding.spec, params)
opt_sharding = jax.tree.map(lambda x: x.sharding.spec, optimizer)

@partial(shard_map, mesh=mesh, in_specs=(param_sharding, opt_sharding, P("data")), out_specs=(param_sharding, opt_sharding, P()), check_rep=False)
def do_update(params, optimizer, tokens_batch):
    loss, grad = fast_batch_grad(params, tokens_batch)
    loss = jax.lax.pmean(loss, "data")
    grad = jax.tree.map(sync_grads, grad, is_leaf=lambda x: isinstance(x, Partitioned))
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
