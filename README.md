# Keras EdgeConv-Layer

General implementation of the EdgeConv-Block as described in [ParticleNet: Jet Tagging via Particle Clouds](https://arxiv.org/abs/1902.08570).

## Using the layer

In order to use the layer class found in [edgeconv.py](edgeconv.py), a h-function needs to be defined. This function should take a Keras tensor of length 2*C with C being the dimension of the features. The h-function might include other Keras layers. Setting the number of k nearest neighbors to be considered using the `next_neighbors` argument, the EdgeConv layer can be implemented as demonstrated in [test.py](test.py).

## Acknowledgement
This implementation borrows code from the ParticleNet tensorflow [implementation](https://github.com/hqucms/ParticleNet).
