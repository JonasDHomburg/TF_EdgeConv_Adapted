# Keras EdgeConv-Layer

General implementation of the EdgeConv-Block as described in [Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).

## Using the layer

In order to use the layer class found in [edgeconv.py](edgeconv.py), a kernel-function needs to be defined. This function should take a list of two Keras tensors of length C with C being the dimension of the features. The kernel-function might must include Keras layers. Setting the number of k nearest neighbors to be considered using the `next_neighbors` argument, the EdgeConv layer can be implemented as demonstrated in [test.py](test.py).

## Acknowledgement
This implementation borrows code from the ParticleNet tensorflow [implementation](https://github.com/hqucms/ParticleNet).
