!-------------------------------------------------------------------------------
! TODO:
!       * expand to full test with all data
!       * ensure deconvolution correctness; this works very well
!         with some random initializations but not others
!           * implement different weight initializers
!       * implement proper "prediction" function for images
!       * add Python helper file to visualize output
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! uses sequential neural network as an autoencoder on MNIST dataset, which
! contains many flattened 28x28x1 grayscale pictures of handwritten digits
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
! compile: make autoenc
! run:     ./autoenc
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

program main
    use net_helper_procedures
    use sequential_neural_net
    implicit none

    class(SeqNN), pointer     :: snn
    real(kind=8), allocatable :: image(:,:,:), train_images(:,:,:,:), &
                                 test_images(:,:,:,:), train(:,:), test(:,:), &
                                 train_x(:,:), test_x(:,:)
    integer                   :: train_rows, test_rows, variables, classes, &
                                 pixels, row, i

    train_rows = 2
    test_rows = 2
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
    ! separate train labels from pixels
    train_x = train(:,2:) / 255 ! scale pixels to [0,1]

    deallocate(train)

    !---------------------------------------------------------------------------
    ! reshape train examples from dimension 1x784 to 28x28x1 

    allocate(train_images(28,28,1,train_rows))

    do i = 1, train_rows
        ! 28x28x1 array filled column-major order
        image = reshape(train_x(i,:), [28,28,1])

        ! transpose gives "expected" row-major order
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
    ! separate test labels from pixels
    test_x = test(:,2:) / 255 ! scale to [0,1]

    deallocate(test)

    !---------------------------------------------------------------------------
    ! reshape test examples from dimension 1x784 to 28x28x1 

    allocate(test_images(28,28,1,test_rows))

    do i = 1, test_rows
        ! 28x28x1 array filled column-major order
        image = reshape(test_x(i,:), [28,28,1])

        ! transpose gives "expected" row-major order
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

    !-----------------------------------------------------
    ! encoder layers
    call snn%snn_add_conv_layer(input_dims  = [28,28,1],&
                                kernels     = 12, &
                                kernel_dims = [5,5], &
                                stride      = [1,1], &
                                activ       = 'relu', &
                                padding     = 'valid')

    call snn%snn_add_conv_layer(kernels     = 12, &
                                kernel_dims = [5,5], &
                                stride      = [1,1], &
                                activ       = 'relu', &
                                padding     = 'valid')

    call snn%snn_add_conv_layer(kernels     = 12, &
                                kernel_dims = [5,5], &
                                stride      = [1,1], &
                                activ       = 'relu', &
                                padding     = 'valid')

    call snn%snn_add_conv_layer(kernels     = 12, &
                                kernel_dims = [6,6], &
                                stride      = [2,2], &
                                activ       = 'relu', &
                                padding     = 'valid')

    !-----------------------------------------------------
    ! decoder layers
    call snn%snn_add_conv_layer(kernels     = 12, &
                                kernel_dims = [6,6], &
                                stride      = [2,2], &
                                activ       = 'relu', &
                                padding     = 'full')

    call snn%snn_add_conv_layer(kernels     = 12, &
                                kernel_dims = [5,5], &
                                stride      = [1,1], &
                                activ       = 'relu', &
                                padding     = 'full')

    call snn%snn_add_conv_layer(kernels     = 12, &
                                kernel_dims = [5,5], &
                                stride      = [1,1], &
                                activ       = 'relu', &
                                padding     = 'full')

    call snn%snn_add_conv_layer(kernels     = 1, &
                                kernel_dims = [5,5], &
                                stride      = [1,1], &
                                activ       = 'relu', &
                                padding     = 'full')

    call snn%snn_summary()

    !---------------------------------------------------------------------------
    ! train network

    call snn%snn_fit(conv_input    = train_images, &
                     target_images = train_images, &
                     batch_size    = 2, &
                     epochs        = 60, &
                     learn_rate    = 0.002, &
                     loss          = 'mse', &
                     verbose       = 2)

    !---------------------------------------------------------------------------
    ! store output (reconstruction) to file to visualize

    ! overwrite file (if it exists), start first row
    call write_array_2D(snn%cnn%output%a(:,:,1,1), &
                            'output-data/autoenc_reco.csv', .false.)

    !---------------------------------------------------------------------------

    deallocate(train_images, test_images)
    call deallocate_snn(snn)
end program
