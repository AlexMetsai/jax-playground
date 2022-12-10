"""
Train a fully connect neural network on MNIST digit dataset.
"""

import jax
from jax import random
from sklearn.datasets import load_digits


jax.config.update('jax_platform_name', 'cpu')  # disable irrelevant gpu warning


def init_weights_and_biases(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_weights(dims, key):
    keys = random.split(key, len(dims))
    return [init_weights_and_biases(m, n, k) for m, n, k in zip(dims[:-1], dims[1:], keys)]


dims = [784, 512, 512, 10]
step_size = 0.01
epochs = 10
batch_size = 128
num_targets = 10
weights = init_network_weights(dims, random.PRNGKey(0))
