"""Implementation of embedding layer with shared weights."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from model import model_utils


class EmbeddingSharedWeights(tf.layers.Layer):
    """Calculates input embeddings and pre-softmax linear with shared weights."""

    def __init__(self, vocab_size, hidden_size):
        super(EmbeddingSharedWeights, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        self.shared_weights = None

    def build(self, _):
        with tf.variable_scope("embedding_and_softmax", reuse=tf.AUTO_REUSE):
            # Create and initialize weights. The random normal initializer was chosen
            # randomly, and works well.
            self.shared_weights = tf.get_variable(
                "weights", [self.vocab_size, self.hidden_size],
                initializer=tf.random_normal_initializer(
                    0., self.hidden_size ** -0.5))

        self.built = True

    def call(self, inputs, **kwargs):
        """Get token embeddings of x.

        Args:
          inputs: An int64 tensor with shape [batch_size, length]

        Returns:
          embeddings: float32 tensor with shape [batch_size, length, embedding_size]
          padding: float32 tensor with shape [batch_size, length] indicating the
            locations of the padding tokens in x.
        """
        with tf.name_scope("embedding"):
            embeddings = tf.gather(self.shared_weights, inputs)

            # Scale embedding by the sqrt of the hidden size
            embeddings *= self.hidden_size ** 0.5

            # Create binary array of size [batch_size, length]
            # where 1 = padding, 0 = not padding
            padding = model_utils.get_padding(inputs)

            # Set all padding embedding values to 0
            embeddings *= tf.expand_dims(1 - padding, -1)

        return embeddings

    def linear(self, x):
        """Computes logits by running x through a linear layer.

        Args:
          x: A float32 tensor with shape [batch_size, length, hidden_size]

        Returns:
          float32 tensor with shape [batch_size, length, vocab_size].
        """
        with tf.name_scope("presoftmax_linear"):
            batch_size = tf.shape(x)[0]
            length = tf.shape(x)[1]

            x = tf.reshape(x, [-1, self.hidden_size])
            logits = tf.matmul(x, self.shared_weights, transpose_b=True)

        return tf.reshape(logits, [batch_size, length, self.vocab_size])
