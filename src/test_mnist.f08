!-------------------------------------------------------------------------------
! uses sequential neural network on MNIST dataset, which contains many
! flattened 28x28x1 grayscale pictures of handwritten digits
!
! CSV version of datset downloaded from:
! https://www.kaggle.com/oddrationale/mnist-in-csv
!
! train dimensions:
!   60000 images (rows),
!   785 variables (column 1 = label, other 784 columns are the 28x28 pixels)
!
! test dimensions:
!   10000 images (rows),
!   785 variables (column 1 = label, other 784 columns are the 28x28 pixels)
!
! labels in range [0,9], representing which digit is depicted,
! pixels in range [0, 255], where 0 = black and 255 = white
!-------------------------------------------------------------------------------
! compile: make mnist
! run:     ./mnist
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

program main
    use net_helper_procedures
    use sequential_neural_net
    implicit none

    class(SeqNN), pointer :: snn
    real, allocatable     :: image(:,:,:), train_images(:,:,:,:), &
                             test_images(:,:,:,:), train(:,:), test(:,:), &
                             train_x(:,:), train_y(:), test_x(:,:), test_y(:), &
                             train_y_onehot(:,:), test_y_onehot(:,:)
    integer               :: train_rows, test_rows, variables, classes, &
                             pixels, row, i
    real                  :: accuracy

    train_rows = 60000
    test_rows = 10000
    variables = 785
    pixels = variables - 1
    classes = 10

    !===========================================================================
    !===========================================================================
    ! data processing
    !===========================================================================
    !===========================================================================

    !---------------------------------------------------------------------------
    ! read train data

    print *, '----------------------'
    print *, 'processing train data:'

    allocate(train(train_rows, variables))

    open(unit=1, form='formatted', file='mnist-in-csv/mnist_train.csv')
    read(unit=1, fmt=*) ! skip header line

    do row = 1, train_rows
        read(unit=1, fmt=*) train(row,:)
    end do

    close(unit=1)

    !---------------------------------------------------------------------------
    ! split train data, one hot encode labels

    ! separate train labels and pixels
    train_x = train(:,2:) / 255 ! scale pixels to [0,1]
    train_y = train(:,1)

    call one_hot_encode_special(train_y, classes, train_y_onehot)

    deallocate(train, train_y)

    !---------------------------------------------------------------------------
    ! reshape train examples from dimension 1x784 to 28x28x1 

    allocate(train_images(28,28,1,train_rows))

    do i = 1, train_rows
        ! 28x28x1 array filled column-major order
        image = reshape(train_x(i,:), [28,28,1])

        ! transpose gives "expected" order
        image(:,:,1) = transpose(image(:,:,1))

        train_images(:,:,:,i) = image
    end do

    deallocate(train_x)

    print *, 'done'
    print *, '----------------------'

    !---------------------------------------------------------------------------
    ! read test data

    print *, '----------------------'
    print *, 'processing test data:'

    allocate(test(test_rows, variables))

    open(unit=2, form='formatted', file='mnist-in-csv/mnist_test.csv')
    read(unit=2, fmt=*) ! skip header line

    do row = 1, test_rows
        read(unit=2, fmt=*) test(row,:)
    end do

    close(unit=2)

    !---------------------------------------------------------------------------
    ! split test data, one hot encode labels

    ! separate test labels and pixels
    test_x = test(:,2:) / 255 ! scale to [0,1]
    test_y = test(:,1)

    call one_hot_encode_special(test_y, classes, test_y_onehot)

    deallocate(test, test_y)

    !---------------------------------------------------------------------------
    ! reshape test examples from dimension 1x784 to 28x28x1 

    allocate(test_images(28,28,1,test_rows))

    do i = 1, test_rows
        ! 28x28x1 array filled column-major order
        image = reshape(test_x(i,:), [28,28,1])

        ! transpose gives "expected" order
        image(:,:,1) = transpose(image(:,:,1))

        test_images(:,:,:,i) = image
    end do

    deallocate(test_x, image)

    print *, 'done'
    print *, '----------------------'

    !===========================================================================
    !===========================================================================
    ! network usage
    !===========================================================================
    !===========================================================================
    
    !---------------------------------------------------------------------------
    ! create network

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

    call snn%snn_add_dense_layer(out_nodes  = 512, &
                                 activation = 'relu')

    call snn%snn_add_dense_layer(out_nodes  = 256, &
                                 activation = 'relu')

    call snn%snn_add_dense_layer(out_nodes  = classes, &
                                 activation = 'softmax')

    call snn%snn_summary()

    !---------------------------------------------------------------------------
    ! train network

    call snn%snn_fit(conv_input   = train_images, &
                     train_labels = train_y_onehot, &
                     batch_size   = 128, &
                     epochs       = 2, &
                     learn_rate   = 0.1, &
                     loss         = 'cross_entropy', &
                     verbose      = 2)

    !---------------------------------------------------------------------------
    ! check network accuracy on test data

    accuracy = snn%snn_one_hot_accuracy(conv_input   = test_images, &
                                        input_labels = test_y_onehot, &
                                        verbose      = 2)
    print *, '----------------------'
    print *, 'testing accuracy:', accuracy
    print *, '----------------------'

    !---------------------------------------------------------------------------

    deallocate(train_images, train_y_onehot, test_images, test_y_onehot)
    call deallocate_snn(snn)
end program