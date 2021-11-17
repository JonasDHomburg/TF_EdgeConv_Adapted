from tensorflow import keras
from tensorflow.keras import layers as lay
import edgeconv
import numpy as np


def f(data):
    d1, d2 = data
    dif = lay.Subtract()([d1, d2])
    x = lay.Concatenate(axis=-1)([d1, dif])
    x = lay.Dense(30)(x)
    x = lay.Dense(20)(x)
    return x


# Test EdgeConv initialization, training, saving and loading
points = lay.Input((10, 6))
feats = lay.Input((10, 6))
a = edgeconv.EdgeConv(f, next_neighbors=3)([points, feats])
y = edgeconv.EdgeConv(f, next_neighbors=3)(a)
out = edgeconv.EdgeConv(f, next_neighbors=3)(y)

model = keras.models.Model([points, feats], out)
model.summary()

model.compile(loss="mse", optimizer=keras.optimizers.Adam())

model.fit([np.ones((300, 10, 6)), np.ones((300, 10, 6))], np.ones((300, 10, 20)), epochs=10)


print("\n------------------------- loading and saving -------------------------\n")

model.save("my_model.h5")
m = keras.models.load_model("my_model.h5", {"EdgeConv": edgeconv.EdgeConv, "SplitLayer": edgeconv.SplitLayer, "mean": keras.backend.mean})
m.summary()
