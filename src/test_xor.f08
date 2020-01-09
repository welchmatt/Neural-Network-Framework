!-------------------------------------------------------------------------------
! uses sequential neural network to learn and predict the 4 cases of the XOR
! function, only utilizing DenseLayers;
! uses one hidden layer with 2 nodes and 1 output node
!-------------------------------------------------------------------------------
! compile: make xor
! run:     ./xor
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

program main
    use net_helper_procedures
    use sequential_neural_net
    implicit none

    class(SeqNN), pointer     :: snn
    real(kind=8), allocatable :: X(:,:), Y(:,:), prediction(:,:)
    real(kind=8)              :: loss

    ! hardcode 4 XOR cases
    allocate(X(4,2), Y(4,1))
    X(:,1) = [0,0,1,1] ! A
    X(:,2) = [0,1,0,1] ! B
    Y(:,1) = [0,1,1,0] ! Y = A ^ B

    !---------------------------------------------------------------------------
    ! create network

    snn => create_snn()

    call snn%snn_add_dense_layer(input_nodes = 2, &
                                 out_nodes   = 2, &
                                 activ       = 'elu')

    call snn%snn_add_dense_layer(out_nodes  = 1, &
                                 activ      = 'elu')

    call snn%snn_summary()

    !---------------------------------------------------------------------------
    ! train network

    call snn%snn_fit(dense_input   = X, &
                     target_labels = Y, &
                     batch_size    = 4, &
                     epochs        = 10000, &
                     learn_rate    = 0.4, &
                     loss          = 'mse', &
                     verbose       = 2, &
                     avg_deltas    = .true.)

    !---------------------------------------------------------------------------
    ! check prediction for each case

    call snn%snn_predict(dense_input = X, &
                         res         = prediction)
    
    print *, '----------------------'
    print *, 'prediction:'
    print *, prediction
    print *, 'actual:'
    print *, Y
    print *, '----------------------'

    !---------------------------------------------------------------------------
    ! check network loss on all predictions

    loss = snn%snn_regression_loss(dense_input   = X, &
                                   target_labels = Y, &
                                   loss          = 'mse')

    print *, '----------------------'
    print *, 'loss:'
    print *, loss
    print *, '----------------------'

    !---------------------------------------------------------------------------

    deallocate(X, Y, prediction)
    call deallocate_snn(snn)
end program