# Neural-Network-Framework
A high-level neural network framework, written in pure Fortran.  It supports convolutional, deconvolutional, pooling, dropout, and dense layers.  It can be used to construct regular CNNs and more complex models like autoencoders (see Tests below).

The following is an example CNN to predict on the MNIST dataset.  The name "snn" stands for "sequential neural network," as in Keras, which is used as the foundation for adding different layers.
```fortran
snn => create_snn()

call snn%snn_add_conv_layer(input_dims  = [28,28,1],&
                            kernels     = 32, &
                            kernel_dims = [3,3], &
                            stride      = [1,1], &
                            activ       = 'relu', &
                            padding     = 'valid')

call snn%snn_add_conv_layer(kernels     = 64, &
                            kernel_dims = [3,3], &
                            stride      = [1,1], &
                            activ       = 'relu', &
                            padding     = 'valid')

call snn%snn_add_pool_layer(kernel_dims = [2,2], &
                            stride      = [2,2], &
                            pool        = 'max', &
                            padding     = 'valid')

call snn%snn_add_dropout_layer(drop_rate = 0.1)
call snn%snn_add_dense_layer(out_nodes   = 128, &
                             activ       = 'relu')

call snn%snn_add_dropout_layer(drop_rate = 0.1)
call snn%snn_add_dense_layer(out_nodes   = classes, &
                             activ       = 'softmax')

call snn%snn_summary()
```

We can then view a summary of our model:
```fortran
call snn%snn_summary()
```
```
 ----------------------
 dimensions:               rows        cols    channels
 -----------
 ConvLayer input:            28          28           1
 ConvLayer output:           26          26          32
 ConvLayer output:           24          24          64
 PoolLayer output:           12          12          64
 -----------
 dimensions:              nodes
 -----------
 DenseLayer input:         9216
 DenseLayer output:         128
 DenseLayer output:          10
 ----------------------
```

Then fit the model to our training data:
```fortran
call snn%snn_fit(conv_input    = train_images, &
                 target_labels = train_y_onehot, &
                 batch_size    = 256, &
                 epochs        = 5, &
                 learn_rate    = 0.1, &
                 loss          = 'cross_entropy', &
                 verbose       = 2)
```

Finally, we can calculate the model's accuracy on our testing data:
```fortran
accuracy = snn%snn_one_hot_accuracy(conv_input   = test_images, &
                                    input_labels = test_y_onehot, &
                                    verbose      = 2)
```

Without much fine-tuning of the model's structure (or other optimizations), the above version consistently achieves a testing accuracy of around 97% after an hour of training.

## Prerequisites
* GFortran 8.3.0 (for Fortran 2008 support)
* GNU Make (to build the system)
* Git LFS (to download data)

## Tests
Two tests are stored in the src folder, alongside the framework files:

### test_mnist.f08
Creates a CNN to learn and predict on the MNIST dataset.  See above for an overview of the structure.

Compile and run with:
```
make mnist
./mnist
```
If the MNIST data in the mnist-in-csv folder did not successfulyl download, ensure that you have Git LFS installed, and then pull the data files with:
```
git lfs pull
```

### test_xor.f08
Creates a dense NN to learn and predict the XOR function, using the classic 2-2-1 node structure:

```fortran
snn => create_snn()

call snn%snn_add_dense_layer(input_nodes = 2, &
                             out_nodes   = 2, &
                             activ       = 'elu')

call snn%snn_add_dense_layer(out_nodes  = 1, &
                             activ      = 'elu')
```

This serves as a raw test of regression, rather than classification.  Although there are many local optima in the XOR function, this model will have near-perfect results every couple of (fast) runs, where each column represents one of the four XOR cases (note that no thresholding has been done on the predictions):
```
 prediction:
   9.99178458E-03   1.00000000       1.00000000       9.99211520E-03

 actual:
   0.00000000       1.00000000       1.00000000       0.00000000    
```

Compile and run with:
```
make xor
./xor
```

### test_autoenc.f08
Currently, this is just a sanity check for basic autoencoder functionality using deconvolutional layers (convolutional layers with "full" padding - will adjust to different layer type wrapper in future).  For now, it creates an autoencoder network:
```
----------------------
 ----------------------
 dimensions:               rows        cols    channels
 -----------
 ConvLayer input:            28          28           1
 ConvLayer output:           24          24          12
 ConvLayer output:           20          20          12
 ConvLayer output:           16          16          12
 ConvLayer output:            6           6          12
 ConvLayer output:           16          16          12
 ConvLayer output:           20          20          12
 ConvLayer output:           24          24          12
 ConvLayer output:           28          28           1
 -----------
 ----------------------
```
Then trains on one image as a sanity check.  See plot_autoenc_example.ipynb in src folder for a visualization of the results!  I will later expand this to a test with all MNIST data.

Compile and run with:
```
make autoenc
./autoenc
```

### Cleanup
All compiled files and executables can be removed with: 
```
make allclean
```

All compiled object and module files can be removed with: 
```
make clean
```
