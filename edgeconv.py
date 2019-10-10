"""Implementation of EdgeConv using arbitrary functions as h"""
import tensorflow as tf
import tensorflow.keras.layers as lay
from tensorflow import keras
from tensorflow.keras import backend as K


class EdgeConv(lay.Layer):

    def __init__(self, h_func, next_neighbors,  **kwargs):
        self.h_func = h_func
        self.K = next_neighbors
        super(EdgeConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        try:
            p_shape, f_shape = input_shape
        except ValueError:
            f_shape = input_shape
        print(f_shape)
        print(f_shape.as_list()[-1])
        x = lay.Input((f_shape.as_list()[-1] * 2,))
        self.h_func = keras.models.Model(x, self.h_func(x))
        super(EdgeConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        try:
            points, features = x
        except TypeError:
            points = features = x
        # distance

        # distance
        D = batch_distance_matrix_general(points, points)  # (N, P, P)
        print(D)
        _, indices = tf.nn.top_k(-D, k=self.K + 1)  # (N, P, K+1)
        indices = indices[:, :, 1:]  # (N, P, K)

        fts = features
        knn_fts = knn(indices, fts)  # (N, P, K, C)
        print(knn_fts)
        print(fts)
        knn_fts_center = tf.tile(tf.expand_dims(fts, axis=2), (1, 1, self.K, 1))  # (N, P, K, C)
        knn_fts = tf.concat([knn_fts_center, tf.subtract(knn_fts, knn_fts_center)], axis=-1)  # (N, P, K, 2*C)
        print("knn_fts", knn_fts)
        print("h_func", self.h_func.get_output_shape_at(-1))
        res = lay.TimeDistributed(lay.TimeDistributed(self.h_func))(knn_fts)  # (N,P,K,C')
        # aggregation
        agg = tf.reduce_mean(res, axis=2)  # (N, P, C')
        return agg

    def compute_output_shape(self, input_shape):
        self.output_shape = self.h_func.get_output_shape_at(-1)
        return self.output_shape


def batch_distance_matrix_general(A, B):
    with tf.name_scope('dmat'):
        r_A = tf.reduce_sum(A * A, axis=2, keepdims=True)
        r_B = tf.reduce_sum(B * B, axis=2, keepdims=True)
        m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
        D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
        return D


def knn(topk_indices, features):
    # topk_indices: (N, P, K)
    # features: (N, P, C)
    with tf.name_scope('knn'):
        k = tf.shape(features)[-1]
        num_points = tf.shape(features)[-2]
        queries_shape = tf.shape(features)
        batch_size = queries_shape[0]
        batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, num_points, k, 1))
        indices = tf.concat([batch_indices, tf.expand_dims(topk_indices, axis=3)], axis=3)  # (N, P, K, 2)
        return tf.gather_nd(features, indices)
