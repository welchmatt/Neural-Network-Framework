Framework source code files:
```
conv_layer_definitions.f08
conv_neural_net.f08
dense_layer_definitions.f08
dense_neural_net.f08
net_helper_procedures.f08
pool_layer_definitions.f08
sequential_neural_net.f08
```

BLAS source code files (from http://www.netlib.org/blas/):
```
dgemm.f
lsame.f
xerbla.f
```

Test source code files:
```
test_autoenc.f08
test_mnist.f08
test_xor.f08
```

See the makefile in the parent folder for examples on how to use the modules in your own code.
