# Keras EdgeConv Layer
Adaptation of the EdgeConv implementation by [Niklas Langner](https://git.rwth-aachen.de/niklas.langner/edgeconv_keras) 
which is a general implementation of the EdgeConv-Block as described in 
[Dynamic Graph CNN for Learning on Point Clouds](https://arxiv.org/abs/1801.07829).

## Using the layer

In order to use the layer class found in [edgeconv.py](edgeconv.py), the KNN layer needs to be used first. The KNN layer
takes one or a list of two tensors. The first Tensor contains the coordinates (D dimensions) of points (P) with shape 
(B, P, D) and the second tensor of shape (B, P, C, X..) contains the points features. The result of the KNN layer are 
stacked K features of the nearest K points concatenated with features of the points itself with a shape of 
(B, P, K, 2C, X..). For multidimensional features the dimension to concatenate the neighbors features with the point 
features the parameter 'stack_axis' can be used.

The output of the KNN-Layer can be fed to an instance of a EdgeConv layer. An EdgeConv layer takes a kernel function for
the concatenated features of a point and its neighbor. The kernel function can also be defined in such a way, that it 
takes two tensors, first the features of a point and second the features of a neighbor. Splitting the layers into two 
tensors can be disabled by 'split' (default: 'True'), the dimension is defined by 'split_axis'. Furthermore, the new K 
features are aggregated by the 'agg_func' (default: 'keras.backend.mean'). 

To get an impression on how to use the EdgeConv layer have a look at [test.py](test.py).

## Acknowledgement
This implementation borrows code from:
- [EdgeConv keras](https://git.rwth-aachen.de/niklas.langner/edgeconv_keras)
- ParticleNet tensorflow [implementation](https://github.com/hqucms/ParticleNet).
