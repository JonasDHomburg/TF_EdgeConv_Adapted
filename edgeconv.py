"""Implementation of EdgeConv using arbitrary functions as h"""
import tensorflow as tf
import tensorflow.keras.layers as lay
from tensorflow import keras
from tensorflow.keras import backend as K


class EdgeConv(lay.Layer):

    def __init__(self, h_func, next_neighbors,  **kwargs):
        self.h_func = h_func
        self.nn = next_neighbors
        super(EdgeConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        x = lay.Input(input_shape)
        self.h_func = keras.models.Model(x, self.h_func(x))
        super(EdgeConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.h_func(x)

    def compute_output_shape(self, input_shape):
        self.output_shape = self.h_func.get_output_shape_at(-1)
        return self.output_shape


def knn(X, k):
    r = K.sum(X * X, axis=1, keepdims=True)  # (N, 1, P_A)
    m = K.batch_dot(K.permute_dimensions(K.transpose(X), (2, 0, 1)), X)  # (N, P_A, P_B)
    D = lay.add([lay.subtract([K.permute_dimensions(K.transpose(r), (2, 0, 1)), 2 * m]), r])
    values, indices = tf.nn.top_k(-D, k=k+1)  # (N, P, K+1)
    indices = K.slice(indices, [0, 0, 1], [X.shape[0], X.shape[2], k])  # (N, P, K)
    return indices
