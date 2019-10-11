from tensorflow import keras
from tensorflow.keras import layers as lay
from edgeconv import EdgeConv
import numpy as np


def f(x):
    x = lay.Dense(30)(x)
    x = lay.Dense(20)(x)
    return x


points = lay.Input((10, 5))
feats = lay.Input((10, 5))
a = EdgeConv(f, next_neighbors=3)([points, feats])
y = EdgeConv(f, next_neighbors=3)(a)
out = EdgeConv(f, next_neighbors=3)(y)

model = keras.models.Model([points, feats], out)
model.summary()

model.compile(loss="mse", optimizer=keras.optimizers.Adam())

model.fit([np.ones((16, 10, 5)), np.ones((16, 10, 5))], np.ones((16, 10, 20)), epochs=10)
