import tensorflow as tf

print(tf.__version__)
from tensorflow import keras

print(keras.__version__)
from tensorflow.keras import layers as lay
import edgeconv
import numpy as np


class LayerCounter:
    counter = 1


def f(data, counter=LayerCounter()):
    d1, d2 = data
    dif = lay.Subtract(name='Kernel_Subtract_'+str(counter.counter))([d1, d2])
    x = lay.Concatenate(axis=-1, name='Kernel_Concat_'+str(counter.counter))([d1, dif])
    x = lay.Dense(30, name='Kernel_Dense1_'+str(counter.counter))(x)
    x = lay.Dense(20, name='Kernel_Dense2_'+str(counter.counter))(x)
    counter.counter += 1
    return x


dims = 10  # Euclidean dimensions for point location
# Building the Network inputs
points = lay.Input((1000, dims))  # 1k points per measurement
features = lay.Input((1000, 6))  # for each point 6 feature dimensions
features_ = features

# first KNN-EdgeConv combination
# construct tensor of the k nearest features
knn_fts = edgeconv.KNN(k=3, stack_axis=-1, name='KNN1')([points, features_])
# convolve the features
features_ = edgeconv.EdgeConv(kernel_func=f, split=True, split_axis=-1, name='EdgeConv1')(knn_fts)

# second KNN-EdgeConv combination
knn_fts = edgeconv.KNN(k=3, stack_axis=-1, name='KNN2')([points, features_])
features_ = edgeconv.EdgeConv(kernel_func=f, split=True, split_axis=-1, name='EdgeConv2')(knn_fts)

# create model with two tensor input, one tensor output
model = keras.models.Model([points, features], features_)
model.summary()

model.compile(loss="mse", optimizer=keras.optimizers.Adam())

# test model fit with dummy data of 300 measurements/'data points' for one epoch
model.fit([np.ones((300, 1000, dims)), np.ones((300, 1000, 6))], np.ones((300, 1000, 20)), epochs=1)

# test saving
print("\n------------------------- saving -------------------------\n")
model.save("my_model.h5")

# test loading
print("\n------------------------- loading ------------------------\n")
m = keras.models.load_model("my_model.h5", {
    "EdgeConv": edgeconv.EdgeConv,
    "KNN": edgeconv.KNN,
    "SplitLayer": edgeconv.SplitLayer,
    "mean": keras.backend.mean
})

# check if model is still the same
print("\n------------------------- evaluation ---------------------\n")
m.summary()

# test if loaded model is still trainable
model.fit([np.ones((300, 1000, dims)), np.ones((300, 1000, 6))], np.ones((300, 1000, 20)), epochs=1)
