"""Implementation of EdgeConv using arbitrary functions as h"""
import tensorflow.keras.layers as lay
from tensorflow import keras


class EdgeConv(lay.Layer):

    def __init__(self, f, **kwargs):
        self.f = f
        super(EdgeConv, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        x = lay.Input(input_shape)
        self.f = keras.models.Model(x, self.f(x))
        super(EdgeConv, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return self.f(x)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], 30)
