import tensorflow as tf
from tensorflow import keras

tf_version = tuple(map(int, tf.__version__.split('.')))
if tf_version > (2, 2, 3):
    if tf_version < (2, 6, 0):
        from tensorflow.python.keras.engine.functional import Functional
    else:
        from keras.engine.functional import Functional


    def check_kernel(func):
        return not isinstance(func, Functional)
else:
    def check_kernel(func):
        return not isinstance(func, keras.models.Model)


class SplitLayer(keras.layers.Layer):
    """Wrap tf.split into a layer.
    """

    def __init__(self, num_or_size_splits, axis=0, num=None, **kwargs):
        super(SplitLayer, self).__init__(**kwargs)
        self.model = None
        self._output_shape = None
        self.num_or_size_splits = num_or_size_splits
        self.axis = axis
        self.num = num

    def get_config(self):
        return {**super(SplitLayer, self).get_config(),
                **{'num_or_size_splits': self.num_or_size_splits,
                   'num': self.num,
                   'axis': self.axis}}

    def build(self, input_shape):
        super(SplitLayer, self).build(input_shape)
        with tf.name_scope(self.name):
            with tf.name_scope('Split'):
                input_layer = keras.layers.Input(input_shape[1:])
                output_layer = tf.split(input_layer,
                                        num_or_size_splits=self.num_or_size_splits,
                                        axis=self.axis,
                                        num=self.num,
                                        name=self.name + '_Split'
                                        )
                self._output_shape = [shape[1:] for shape in output_layer]
                self.model = keras.models.Model(input_layer, output_layer)

    def call(self, inputs, **kwargs):
        return self.model(inputs)

    def compute_output_shape(self, input_shape):
        return self.output_shape

    def compute_mask(self, inputs, mask=None):
        if mask is None:
            return None
        return tf.split(mask,
                        num_or_size_splits=self.num_or_size_splits,
                        axis=self.axis,
                        num=self.num)

    def __call__(self, *args, **kwargs):
        return super(SplitLayer, self).__call__(*args, **kwargs)


class KNN(keras.layers.Layer):
    """Estimates the nearest K points P' for every point P and stacks the concatenated features of P' and P along a given axis.

    The KNN layer takes one or a list of two tensors. The first Tensor contains the coordinates (D dimensions) of
    points (P) with shape (B, P, D) and the second tensor of shape (B, P, C, X..) contains the points features. The
    result of the KNN layer are stacked K features of the nearest K points concatenated with features of the points
    itself with a shape of (B, P, K, 2C, X..). For multidimensional features the dimension of stacking can be
    defined by 'stack_axis'.

    Example:
    points:
        [[
         [1,],
         [2,],
         [3,],
         [4,],
         [5,]
         ]]
    features:
        [[
         [3,1],
         [4,5],
         [2,7],
         [9,1],
         [1,5]
         ]]

    with k=2:
        nearest neighbors:
            [[1,2],
             [0,2],
             [1,3],
             [2,4],
             [3,2]]
        result:
            [[
             [[3,1, 4,5], [3,1, 2,7]],
             [[4,5, 3,1], [4,5, 2,7]],
             [[2,7, 4,5], [2,7, 9,1]],
             [[9,1, 2,7], [9,1, 1,5]],
             [[1,5, 9,1], [1,5, 2,7]]
             ]]

    Parameters
    ----------
    k Number of nearest neighbors
    stack_axis Axis to stack the features
    """

    @staticmethod
    def batch_distance_matrix_general_large_d(a, b):
        """ Calculate elements-wise distance between entries in two tensors for more than 4 dimensions"""
        with tf.name_scope('dist-mat'):
            r_a = tf.reduce_sum(a * a, axis=2, keepdims=True)
            r_b = tf.reduce_sum(b * b, axis=2, keepdims=True)
            m = tf.matmul(a, tf.transpose(b, perm=(0, 2, 1)))
            return r_a - 2 * m + tf.transpose(r_b, perm=(0, 2, 1))

    @staticmethod
    def batch_distance_matrix_general_small_d(a, b):
        """ Calculate elements-wise distance between entries in two tensors for 4 and fewer dimensions"""
        with tf.name_scope('dist-mat'):
            a_ = tf.expand_dims(a, axis=-1)
            b_ = tf.expand_dims(tf.transpose(b, perm=(0, 2, 1)), axis=1)
            a_b = a_ - b_
            a_b = tf.math.square(a_b)
            return tf.reduce_sum(a_b, axis=2, keepdims=False)

    @staticmethod
    def split_input_one(inputs):
        return [inputs, inputs]

    @staticmethod
    def split_input_two(inputs):
        return inputs

    @staticmethod
    def knn(top_k_indices, features):
        with tf.name_scope('k-features'):
            return tf.gather(params=features, indices=top_k_indices, batch_dims=1)

    def __init__(self, k=3, stack_axis=None, **kwargs):
        super(KNN, self).__init__(**kwargs)
        self.k = k
        self.stack_axis = stack_axis
        if self.stack_axis is None:
            self.stack_axis = -1
        self.feature_dims = None
        self.dist_function = None
        self.split_input = None

    def get_config(self):
        return {**super(KNN, self).get_config(),
                **{'k': self.k, 'stack_axis': self.stack_axis}}

    def build(self, input_shape):
        super(KNN, self).build(input_shape)
        with tf.name_scope(self.name):
            try:
                p_shape, f_shape = input_shape
                self.split_input = self.split_input_two
            except ValueError:
                p_shape = f_shape = input_shape
                self.split_input = self.split_input_one
            self.feature_dims = [1, ] * (len(f_shape) + 1)
            self.feature_dims[2] = self.k
            self.feature_dims = tuple(self.feature_dims)
            if p_shape[-1] > 4:
                self.dist_function = self.batch_distance_matrix_general_large_d
            else:
                self.dist_function = self.batch_distance_matrix_general_small_d

    def call(self, inputs, **kwargs):
        with tf.name_scope(self.name):
            points, features = self.split_input(inputs)
            dist_mat = self.dist_function(points, points)  # (N, P, P)
            _, indices = tf.nn.top_k(-dist_mat, k=self.k + 1)  # (N, P, K+1)
            indices = indices[:, :, 1:]  # (N, P, K)
            knn_fts = self.knn(indices, features)  # (N, P, K, C, ..)
            knn_fts_center = tf.tile(tf.expand_dims(features, axis=2), self.feature_dims)  # (N, P, K, C, ..)
            knn_fts = tf.concat([knn_fts_center, knn_fts], axis=self.stack_axis)  # (N, P, K, 2*C, ..)
            return knn_fts

    def compute_output_shape(self, input_shape):
        try:
            _, f_shape = input_shape
        except ValueError:
            f_shape = input_shape
        f_shape = f_shape.as_list()
        f_shape[self.stack_axis] = f_shape[self.stack_axis] * 2
        return f_shape

    def __call__(self, *args, **kwargs):
        return super(KNN, self).__call__(*args, **kwargs)


class EdgeConv(keras.layers.Layer):
    """
    Keras layer implementation of EdgeConv.
    # Arguments
        kernel_func: h-function applied on the points, and it's k nearest neighbors. The function can take one or two
            arguments. If two tensors are provided, the first tensor is the vector v_i of the central point, the second
            tensor is the vector of one of its neighbors v_j. Otherwise, those two tensors are stacked along an axis.
            :param list: [v_i, v_j] with v_i and v_j being Keras tensors with shape (C_f, ..) or v_ij being Keras with
                shape (2*C_f, ..).
            :return: Keras tensor of shape (C', ).
        agg_func: Aggregation function applied after h. Must take argument "axis=aggregate_axis" to
            aggregate over all neighbors.
        split: The input tensor can be split at given `split_axis` if True.
        split_axis: The axis to split the input tensor before the result is fed to the kernel_func.
        aggregate_axis: The axis to aggregate the result over all neighbors.
    # Input shape
        One tensor with shape:
         `(N, P, K, 2*C, ..)`.
    # Output shape
        Tensor with shape:
        `(N, P, C_h, ..)`
        with (C_h, ..) being the output dimension of the h-function.
    """

    def __init__(self, kernel_func, agg_func=keras.backend.mean, split=True, split_axis=None, aggregate_axis=2,
                 **kwargs):
        super(EdgeConv, self).__init__(**kwargs)
        self.kernel_func = kernel_func
        self.agg_func = agg_func
        self.split = split
        self.split_axis = split_axis
        self.aggregate_axis = aggregate_axis
        self.distributed_kernel = None
        if self.split_axis is None:
            self.split_axis = -1
        if type(agg_func) == str:
            raise ValueError(
                "No such agg_func '%s'. When loading the model specify the agg_func '%s' via custom_objects" % (
                    agg_func, agg_func))

    def get_config(self):
        return {**super(EdgeConv, self).get_config(),
                **{'kernel_func': self.kernel_func,
                   'agg_func': self.agg_func,
                   'split': self.split,
                   'split_axis': self.split_axis,
                   'aggregate_axis': self.aggregate_axis}}

    def build(self, input_shape):
        super(EdgeConv, self).build(input_shape)
        with tf.name_scope(self.name):
            if check_kernel(self.kernel_func):  # for not wrapping model around model when loading model
                with tf.name_scope('EdgeConv-Kernel'):
                    in_shape = input_shape.as_list()[3:]  # (C, ..)
                    x = keras.layers.Input(tuple(in_shape))
                    if self.split:
                        x1, x2 = SplitLayer(
                            num_or_size_splits=2,
                            axis=self.split_axis,
                            name=self.name + '_SplitLayer',
                        )(x)
                        y = self.kernel_func([x1, x2])
                    else:
                        y = self.kernel_func(x)
                    self.kernel_func = keras.models.Model(x, y)
            self.distributed_kernel = keras.layers.TimeDistributed(keras.layers.TimeDistributed(
                self.kernel_func,
                name=self.name + '_Nodes'),
                name=self.name + '_Batch')

    def call(self, inputs, **kwargs):
        with tf.name_scope(self.name):
            with tf.name_scope('EdgeConv'):
                conved = self.distributed_kernel(inputs)
                aggregated = self.agg_func(conved, axis=self.aggregate_axis)
                return aggregated

    def compute_output_shape(self, input_shape):  # (N, P, K, 2*C, ..)
        output_shape = input_shape.as_list()[:3] + self.kernel_func.output_shape.as_list()
        return output_shape

    def __call__(self, *args, **kwargs):
        return super(EdgeConv, self).__call__(*args, **kwargs)
