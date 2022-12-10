"""
Train a fully connect neural network on MNIST digit dataset.
"""

import jax
import jax.numpy as jnp
from jax import random, vmap
from jax.scipy.special import logsumexp
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


def relu(x):
    return jnp.maximum(0, x)


def predict(weights, image):
    """Defining for a single sample, batches are handled with jax.vamp"""
    activation = image
    for w, b in weights[:-1]:
        output = jnp.dot(w, activation) + b
        activation = relu(output)

    last_layer_w, _last_layer_b = weights[-1]
    logits = jnp.dot(last_layer_w, activation) + _last_layer_b
    return logits - logsumexp(logits)


batch_predict = vmap(predict, in_axes=(None, 0))


def data_loader(batch_size):
    """
    Both TensorFlow and PyTorch offer superb data loaders, but I don't want
    to rely on these packages for the moment, hence the custom data loader.
    """
    mnist_data = load_digits()
    images_flat, labels = mnist_data["data"], mnist_data["target"]

    # padding for last batch
    pad_len = batch_size - len(images_flat) % batch_size
    images_flat = jnp.concatenate((images_flat, images_flat[:pad_len]))
    labels = jnp.concatenate((labels, labels[:pad_len]))

    # arrange in batches
    num_batches = int(len(images_flat) / batch_size)
    images_flat = images_flat.reshape(batch_size, num_batches, -1)
    labels = labels.reshape(batch_size, num_batches, -1)

    for imgs, labels in zip(images_flat, labels):
        yield imgs, labels
