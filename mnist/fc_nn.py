"""
Train a fully connect neural network on MNIST digit dataset.
"""

import jax
import jax.numpy as jnp
import mnist
from jax import grad, jit, vmap, random
from jax.scipy.special import logsumexp


jax.config.update('jax_platform_name', 'cpu')  # disable irrelevant gpu warning


def init_weights_and_biases(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def init_network_weights(dims, key):
    keys = random.split(key, len(dims))
    return [init_weights_and_biases(m, n, k) for m, n, k in zip(dims[:-1], dims[1:], keys)]


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


def one_hot(x, bits, dtype=jnp.float32):
    return jnp.array(x[:, None] == jnp.arange(bits), dtype)


def accuracy(weights, images_flat, labels):
    target_class = jnp.argmax(labels, axis=1)
    predicted_class = jnp.argmax(batch_predict(weights, images_flat), axis=1)
    return jnp.mean(predicted_class == target_class)


def loss(weights, images_flat, targets):
    preds = batch_predict(weights, images_flat)
    return -jnp.mean(preds * targets)


@jit
def update(weights, x, y):
    grads = grad(loss)(weights, x, y)
    return [(w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(weights, grads)]


def data_loader(images_flat, labels, batch_size):
    """
    Both TensorFlow and PyTorch offer superb data loaders, but I don't want
    to rely on these packages for the moment, hence the custom data loader.
    """

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


dims = [784, 512, 512, 10]
step_size = 0.01
epochs = 10
batch_size = 128
num_targets = 10
weights = init_network_weights(dims, random.PRNGKey(0))

# Load data
train_images = mnist.train_images()
train_images = train_images.reshape(train_images.shape[0], -1)
train_labels = one_hot(mnist.train_labels(), 10)
test_images = mnist.test_images()
test_images = test_images.reshape(test_images.shape[0], -1)
test_labels = one_hot(mnist.test_labels(), 10)

# Train model
for epoch in range(epochs):
    for x, y in data_loader(train_images, train_labels, batch_size=batch_size):
        weights = update(weights, x, y)
        temp = weights
    train_acc = accuracy(weights, train_images, train_labels)
    test_acc = accuracy(weights, test_images, test_labels)
    print(f"Training accuracy: {train_acc}")
    print(f"Testing accruacy: {test_acc}")
