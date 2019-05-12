!-------------------------------------------------------------------------------
! TODO:
!   * update snn_predict to handle only ConvNN present (need two functions)
!-------------------------------------------------------------------------------

!-------------------------------------------------------------------------------
! sequential neural network implementation that utilizes ConvLayers, PoolLayers,
! and DenseLayers in "sequence" (hence the name sequential - based on Keras)
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module sequential_neural_net
use net_helper_procedures
use conv_layer_definitions
use dense_layer_definitions
use conv_neural_net
use dense_neural_net
implicit none

!===============================================================================
!===============================================================================
! procedures with 2D array input require variables-as-columns form
!===============================================================================
!===============================================================================

!===============================================================================
! types
!===============================================================================

! represents a neural network with support for ConvLayers (with PoolLayers)
! followed by DenseLayers 'in sequence', or only DenseLayers
!
! serves as a wrapper for a ConvNN for ConvLayers and PoolLayers, followed by a
! DenseNN for DenseLayers
!
! if both networks are present, this function handles the transfer of data from
! ConvNN through DenseNN in forward propagation, and the opposite way for
! backpropagation
!
! ConvLayers and PoolLayers wrapped by ConvNN, DenseLayers wrapped by DenseNN
type :: SeqNN
    class(ConvNN), pointer  :: cnn
    class(DenseNN), pointer :: dnn
    logical                 :: cnn_done, is_init ! mutex locks for creation
    integer                 :: batch_size
contains
    ! wrapper functions that initiate ConvNN/DenseNN traversals, and also handle
    ! transferring data between them. includes higher level proceduers for
    ! testing and training the overall network
    procedure, pass         :: snn_add_conv_layer, snn_add_pool_layer, &
                               snn_add_dense_layer, snn_init, snn_forw_prop, &
                               snn_cnn_out_delta, snn_back_prop, snn_update, &
                               snn_fit, snn_one_hot_accuracy, &
                               snn_regression_loss, snn_predict, &
                               snn_summary
end type
contains

!===============================================================================
! constructors / destructors
!===============================================================================

!-------------------------------------------------------------------------------
! constructs a new SeqNN
!-------------------------------------------------------------------------------
! 
!-------------------------------------------------------------------------------
! returns :: (SeqNN pointer) new SeqNN
!-------------------------------------------------------------------------------
function create_snn()
    class(SeqNN), pointer :: create_snn

    allocate(create_snn)
    create_snn%is_init =  .false.
    create_snn%cnn     => null()
    create_snn%dnn     => null()

    ! can add ConvLayers until first DenseLayer added;
    ! cannot complete snn until a DenseLayer is added
    create_snn%cnn_done = .false.
end function

!-------------------------------------------------------------------------------
! deallocate a SeqNN
!-------------------------------------------------------------------------------
! snn:      (SeqNN pointer)
!-------------------------------------------------------------------------------
! alters :: snn and the ConvNN and DenseNN it wraps are deallocated
!-------------------------------------------------------------------------------
subroutine deallocate_snn(snn)
    class(SeqNN), pointer :: snn

    if (snn%is_init) then
        if (associated(snn%cnn)) then
            call deallocate_cnn(snn%cnn)
        end if

        if (associated(snn%dnn)) then
            call deallocate_dnn(snn%dnn)
        end if
        deallocate(snn)
    end if
end subroutine

!===============================================================================
! SeqNN procedures
!   *** all require input and labels in variables-as-columns form
!===============================================================================

!-------------------------------------------------------------------------------
! create and add a new ConvLayer to tail of this SeqNN's ConvNN linked list
!
! input_dims must be specified to first call to this subroutine
!-------------------------------------------------------------------------------
! this:        (SeqNN - implicitly passed)
! kernels:     (integer) kernels in new layer
! kernel_dims: (integer(2)) (height, width) of each kernel channel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! activ:       (characters) activation function
! padding:     (characters) padding type
!
! input_dims:  (optional - integer(3)) (height, width, channels) of one input
! drop_rate:   (optional - real) % of input nodes to dropout
!-------------------------------------------------------------------------------
! alters ::    new ConvLayer appended to this SeqNN's ConvNN linked list
!-------------------------------------------------------------------------------
subroutine snn_add_conv_layer(this, kernels, kernel_dims, stride, activ, &
                              padding, input_dims, drop_rate)
    class(SeqNN)                       :: this
    integer, intent(in)                :: kernels, kernel_dims(2), stride(2)
    character(*), intent(in)           :: activ, padding
    integer, intent(in), optional      :: input_dims(3)
    real, intent(in), optional         :: drop_rate
    real                               :: drop

    if (.not. (activ == 'sigmoid' .or. activ == 'relu' .or. &
        activ == 'leaky_relu' .or. activ == 'elu')) then
        print *, '---------------------------------------------'
        print *, '(sequential_neural_net :: snn_add_conv_layer)'
        print *, 'invalid activation function.'
        print *, 'supported: sigmoid, relu, leaky_relu, elu'
        print *, '---------------------------------------------'
        stop -1
    end if

    if (this%cnn_done) then
        print *, '---------------------------------------------'
        print *, '(sequential_neural_net :: snn_add_conv_layer)'
        print *, 'cannot add new ConvLayer.'
        print *, '---------------------------------------------'
        stop -1
    end if

    if (.not. associated(this%cnn)) then
        ! creating new cnn; requires input_dims
        if (present(input_dims)) then
            this%cnn => create_cnn(input_dims)
        else
            print *, '---------------------------------------------'
            print *, '(sequential_neural_net :: snn_add_conv_layer)'
            print *, 'must supply input_dims for first ConvLayer.'
            print *, '---------------------------------------------'
            stop -1
        end if
    else
        ! cnn already exists; do not allow input_dims
        if (present(input_dims)) then
            print *, '---------------------------------------------'
            print *, '(sequential_neural_net :: snn_add_conv_layer)'
            print *, 'only pass input_dims to first ConvLayer.'
            print *, '---------------------------------------------'
            stop -1
        end if
    end if

    if (present(drop_rate)) then
        drop = drop_rate
    else
        drop = 0
    end if

    call this%cnn%cnn_add_layer(kernels, kernel_dims, stride, &
                                activ, padding, drop)
end subroutine

!-------------------------------------------------------------------------------
! create and add a new PoolLayer to tail of this SeqNN's ConvNN linked list
!-------------------------------------------------------------------------------
! this:        (SeqNN - implicitly passed)
! kernel_dims: (integer(2)) (height, width) of pool kernel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! pool:        (characters) pool type
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! alters ::    new ConvLayer appended to this SeqNN's ConvNN linked list
!-------------------------------------------------------------------------------
subroutine snn_add_pool_layer(this, kernel_dims, stride, pool, padding)
    class(SeqNN)             :: this
    integer, intent(in)      :: kernel_dims(2), stride(2)
    character(*), intent(in) :: pool, padding

    if (this%cnn_done) then
        print *, '---------------------------------------------'
        print *, '(sequential_neural_net :: snn_add_pool_layer)'
        print *, 'cannot add new PoolLayer.'
        print *, '---------------------------------------------'
        stop -1
    end if

    if (.not. associated(this%cnn)) then
        print *, '---------------------------------------------'
        print *, '(sequential_neural_net :: snn_add_pool_layer)'
        print *, 'PoolLayer must follow ConvLayer.'
        print *, '---------------------------------------------'
        stop -1
    end if

    call this%cnn%cnn_add_pool_layer(kernel_dims, stride, pool, padding)
end subroutine

!-------------------------------------------------------------------------------
! create and add a new DenseLayer to tail of this SeqNN's DenseNN linked list
!
! input_nodes must be specified to first call to this subroutine if NO
! ConvLayers added before it (in which case, SeqNN is a pure DenseNN)
!-------------------------------------------------------------------------------
! this:        (SeqNN - implicitly passed)
! out_nodes:   (integer) nodes output by new DenseLayer
! activ:       (characters) activation function
!
! input_nodes: (optional - integer) input variables
! drop_rate:   (optional - real) % of input nodes to dropout
!-------------------------------------------------------------------------------
! alters ::    new DenseLayer appended to this SeqNN's DenseNN linked list
!-------------------------------------------------------------------------------
subroutine snn_add_dense_layer(this, out_nodes, activ, input_nodes, drop_rate)
    class(SeqNN)                       :: this
    integer, intent(in)                :: out_nodes
    character(*), intent(in)           :: activ
    integer, intent(in), optional      :: input_nodes
    real, intent(in), optional         :: drop_rate
    real                               :: drop

    if (.not. (activ == 'sigmoid' .or. activ == 'relu' .or. &
        activ == 'leaky_relu' .or. activ == 'elu' .or. &
        activ == 'softmax')) then
        print *, '---------------------------------------------'
        print *, '(sequential_neural_net :: snn_add_conv_layer)'
        print *, 'invalid activation function.'
        print *, 'supported: sigmoid, relu, leaky_relu, elu'
        print *, 'also supported for output layer: softmax'
        print *, '---------------------------------------------'
        stop -1
    end if

    if (.not. this%cnn_done) then
        ! do not allow adding more ConvLayers after this new DenseLayers
        this%cnn_done = .true.

        if (associated(this%cnn)) then
            ! create dnn "appended" to cnn (if one exists); input_nodes from cnn
            if (present(input_nodes)) then
                print *, '----------------------------------------------'
                print *, '(sequential_neural_net :: snn_add_dense_layer)'
                print *, 'only pass input_dims to first DenseLayer.'
                print *, '----------------------------------------------'
                stop -1
            end if

            ! each node in cnn output gets node in dnn
            this%dnn => create_dnn(this%cnn%out_count)
        else
            ! no cnn, create dnn on its own; requires input_nodes
            if (present(input_nodes)) then
                this%dnn => create_dnn(input_nodes)
            else
                print *, '----------------------------------------------'
                print *, '(sequential_neural_net :: snn_add_dense_layer)'
                print *, 'must supply input_nodes for first DenseLayer.'
                print *, '----------------------------------------------'
                stop -1
            end if
        endif
    else
        ! not creating the dnn this time; do not allow input_nodes
        if (present(input_nodes)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_add_dense_layer)'
            print *, 'only pass input_dims to first DenseLayer.'
            print *, '----------------------------------------------'
            stop -1
        end if
    end if

    if (present(drop_rate)) then
        drop = drop_rate
    else
        drop = 0
    end if

    call this%dnn%dnn_add_layer(out_nodes, activ, drop)
end subroutine

!-------------------------------------------------------------------------------
! initialize this SeqNN's ConvNN and DenseNN, and specify batch size to process
!-------------------------------------------------------------------------------
! this:       (SeqNN - implicitly passed)
! batch_size: (integer) examples to process before back prop
!-------------------------------------------------------------------------------
! alters ::   this SeqNN's ConvNN and DenseNN are alloacted and become usable
!-------------------------------------------------------------------------------
subroutine snn_init(this, batch_size)
    class(SeqNN)        :: this
    integer, intent(in) :: batch_size

    this%is_init = .true.
    this%batch_size = batch_size

    if (associated(this%cnn)) then
        call this%cnn%cnn_init(batch_size)
    end if

    if (associated(this%dnn)) then
        call this%dnn%dnn_init(batch_size)
    else
        this%cnn_done = .true. ! complete the existing cnn
    end if
end subroutine

!-------------------------------------------------------------------------------
! wrapper to forward propagate input through SeqNN's ConvNN then DenseNN
!
! must only pass conv_batch if ConvLayers in SeqNN, otherwise
! must only pass dense_batch if no ConvLayers present
!-------------------------------------------------------------------------------
! this:        (SeqNN - implicitly passed)
! is_train:    (logical) in training iteration
!
! conv_batch:  (optional - real(:,:,:,:)) input batch for ConvLayers
! dense_batch: (optional - real(:,:)) input batch for DenseLayers
!-------------------------------------------------------------------------------
! alters ::    this SeqNN's ConvNN and DenseNN layers' z's and a's calculated
!-------------------------------------------------------------------------------
subroutine snn_forw_prop(this, is_train, conv_batch, dense_batch)
    class(SeqNN)                       :: this
    real(kind=8), intent(in), optional :: conv_batch(:,:,:,:), dense_batch(:,:)
    logical, intent(in)                :: is_train
    real(kind=8), allocatable          :: dnn_batch(:,:)
    integer                            :: i

    if (associated(this%cnn)) then
        ! cnn present; must only pass conv_batch
        if (.not. present(conv_batch) .or. present(dense_batch)) then
            print *, '-----------------------------------------'
            print *, '(sequential_neural_net :: snn_forw_prop)'
            print *, 'ConvLayers present: only pass conv_batch.'
            print *, '-----------------------------------------'
            stop -1
        end if

        call this%cnn%cnn_forw_prop(conv_batch, is_train)

        ! stop here if no dnn to feed into
        if (.not. associated(this%dnn)) then
            return
        end if

        ! prepare cnn output to pass into dnn:
        ! flatten each cnn output prediction (3D) in batch into rows for dnn
        allocate(dnn_batch(this%batch_size, this%cnn%out_count))

        do i = 1, this%batch_size
            ! store flattened prediction rows; check if output pooled
            if (associated(this%cnn%output%next_pool)) then
                ! pooled shape
                dnn_batch(i,:) = reshape(this%cnn%output%next_pool%a(:,:,:,i), &
                                         [this%cnn%output%next_pool%out_count])
            else
                ! regular shape (no pool)
                dnn_batch(i,:) = reshape(this%cnn%output%a(:,:,:,i), &
                                         [this%cnn%output%out_count])
            end if
        end do
    else if (associated(this%dnn)) then
        ! dnn but cnn not present; must only pass dense_batch
        if (.not. present(dense_batch) .or. present(conv_batch)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_forw_prop)'
            print *, 'ConvLayers not present: only pass dense_batch.'
            print *, '----------------------------------------------'
            stop -1
        end if

        dnn_batch = dense_batch
    end if

    ! forward prop through dnn (either from cnn or direct input)
    if (associated(this%dnn)) then
        call this%dnn%dnn_forw_prop(dnn_batch, is_train)
    end if
end subroutine

!-------------------------------------------------------------------------------
! once deltas calculated for DenseLayers from snn_back_prop, calculate
! derivative of cost wrt output of final ConvLayer; this connects DenseNN to
! ConvNN in backpropagation
!-------------------------------------------------------------------------------
! this:     (SeqNN - implicitly passed)
!-------------------------------------------------------------------------------
! alters :: this SeqNN's ConvNN output ConvLayer's d is calculated
!-------------------------------------------------------------------------------
subroutine snn_cnn_out_delta(this)
    class(SeqNN)              :: this
    real(kind=8), allocatable :: d_wrt_a(:,:), d_slice(:,:,:)
    integer                   :: i

    ! first find derivative of cost wrt a(L) OR pool(L) (if present);
    ! each row in d_wrt_a is an entry in the batch:
    ! d_wrt_a = matmul(this%dnn%first_hid%d, transpose(this%dnn%first_hid%w))
    call dgemm_wrapper(this%dnn%first_hid%d, this%dnn%first_hid%w, d_wrt_a, &
                       transb=.true.)

    this%cnn%output%d = 0

    do i = 1, this%batch_size
        if (associated(this%cnn%output%next_pool)) then
            ! first reshape deltas into shape of pool output
            d_slice = reshape(d_wrt_a(i,:), &
                      shape(this%cnn%output%next_pool%a(:,:,:,i)))

            ! must undo the pooling to find derivative wrt a's kept by pool
            call this%cnn%output%next_pool%pool_back_prop(d_slice, i, &
                                                          this%cnn%output%d)
        else
            ! no pooling to account for; reshape back to regular cnn dimensions
            this%cnn%output%d(:,:,:,i) = reshape(d_wrt_a(i,:), &
                                            shape(this%cnn%output%d(:,:,:,i)))
        end if
    end do

    ! derivative of loss wrt cnn%output%z
    this%cnn%output%d = this%cnn%output%d * &
                        activfunc_deriv(this%cnn%output%z, &
                                        this%cnn%output%activ)
end subroutine

!-------------------------------------------------------------------------------
! wrapper to back propagate through SeqNN's DenseNN then ConvNN
!-------------------------------------------------------------------------------
! this:     (SeqNN - implicitly passed)
! loss:     (characters) loss function
!
! target_labels:   (optional - real(:,:)) labels we are trying to predict
! target_images:   (optional - real(:,:,:,:)) images we are trying to predict
!-------------------------------------------------------------------------------
! alters :: this SeqNN's ConvNN and DenseNN layers' d's are calculated
!-------------------------------------------------------------------------------
subroutine snn_back_prop(this, loss, target_labels, target_images)
    class(SeqNN)                       :: this
    character(*), intent(in)           :: loss
    real(kind=8), intent(in), optional :: target_labels(:,:), &
                                          target_images(:,:,:,:)
    logical                            :: out_delta_done

    out_delta_done = .false.

    if (associated(this%dnn)) then
        if (.not. present(target_labels) .or. present(target_images)) then
            print *, '-------------------------------------------'
            print *, '(sequential_neural_net :: snn_back_prop)'
            print *, 'DenseLayer output: only pass target_labels.'
            print *, '-------------------------------------------'
            stop -1
        end if

        call this%dnn%dnn_back_prop(target_labels, loss)

        if (associated(this%cnn)) then
            call this%snn_cnn_out_delta() ! transfer delta to cnn
            out_delta_done = .true.
        end if
    end if

    ! continue backprop through cnn
    if (associated(this%cnn)) then
        if (out_delta_done) then
            ! fed by dnn
            call this%cnn%cnn_back_prop(out_delta_done)
        else
            ! not fed by dnn; find loss directly
            if (.not. present(target_images) .or. present(target_labels)) then
                print *, '-----------------------------------------'
                print *, '(sequential_neural_net :: snn_back_prop)'
                print *, 'ConvLayer output: only pass target_images.'
                print *, '-----------------------------------------'
                stop -1
            end if

            call this%cnn%cnn_back_prop(out_delta_done, target_images, loss)
        end if
    end if
end subroutine

!-------------------------------------------------------------------------------
! wrapper to adjust kernels, weights, biases in this SeqNN's ConvNN and DenseNN
!
! must only pass conv_batch if ConvLayers in SeqNN, otherwise
! must only pass dense_batch if no ConvLayers present
!-------------------------------------------------------------------------------
! this:        (SeqNN - implicitly passed)
! learn_rate:  (real) scale factor for change in kernels and biases
! is_train:    (logical) in training iteration
!
! conv_batch:  (optional - real(:,:,:,:)) input batch for ConvLayers
! dense_batch: (optional - real(:,:)) input batch for DenseLayers
!-------------------------------------------------------------------------------
! alters ::    this SeqNN's kernels, weights, biases adjusted to minimize loss
!-------------------------------------------------------------------------------
subroutine snn_update(this, learn_rate, is_train, conv_batch, dense_batch)
    class(SeqNN)                       :: this
    real, intent(in)                   :: learn_rate
    real(kind=8), intent(in), optional :: conv_batch(:,:,:,:), dense_batch(:,:)
    logical, intent(in)                :: is_train
    real(kind=8), allocatable          :: dnn_batch(:,:)
    integer                            :: i

    if (associated(this%cnn)) then
        ! cnn present; must only pass conv_batch
        if (.not. present(conv_batch) .or. present(dense_batch)) then
            print *, '-----------------------------------------'
            print *, '(sequential_neural_net :: snn_update)'
            print *, 'ConvLayers present: only pass conv_batch.'
            print *, '-----------------------------------------'
            stop -1
        end if

        call this%cnn%cnn_update(conv_batch, learn_rate, is_train)

        ! stop here if no dnn to feed into
        if (.not. associated(this%dnn)) then
            return
        end if

        ! prepare cnn output to pass into dnn:
        ! flatten each cnn output prediction (3D) in batch into rows for dnn
        allocate(dnn_batch(this%batch_size, this%cnn%out_count))

        do i = 1, this%batch_size
            ! store flattened prediction rows
            if (associated(this%cnn%output%next_pool)) then
                ! pooled shape
                dnn_batch(i,:) = reshape(this%cnn%output%next_pool%a(:,:,:,i), &
                                         [this%cnn%output%next_pool%out_count])
            else
                ! regular shape (no pool)
                dnn_batch(i,:) = reshape(this%cnn%output%a(:,:,:,i), &
                                         [this%cnn%output%out_count])
            end if
        end do
    else
        ! cnn not present; must only pass dense_batch
        if (.not. present(dense_batch) .or. present(conv_batch)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_update)'
            print *, 'ConvLayers not present: only pass dense_batch.'
            print *, '----------------------------------------------'
            stop -1
        end if

        dnn_batch = dense_batch
    end if

    ! update through dnn (either from cnn or direct input)
    if (associated(this%dnn)) then
        call this%dnn%dnn_update(dnn_batch, learn_rate, is_train)
    end if
end subroutine

!-------------------------------------------------------------------------------
! handles training SeqNN on all training data (inputs, labels)
!
! must only pass conv_input if ConvLayers in SeqNN, otherwise
! must only pass dense_input if no ConvLayers present
!-------------------------------------------------------------------------------
! this:          (SeqNN - implicitly passed)
! batch_size:    (integer) examples to process before back prop
! epochs:        (integer) how many times to pass over all the training data
! learn_rate:    (real) scale factor for change in kernels and biases
! loss:          (characters) loss function
!
! conv_input:    (optional - real(:,:,:,:)) input for ConvLayers
! dense_input:   (optional - real(:,:)) input for DenseLayers
! target_labels: (optional - real(:,:)) all targets we are trying to predict
! target_images: (optional - real(:,:,:,:)) all images we are trying to predict
! verbose:       (optional - integer) 0 = none, 1 = epochs, 2 = 1 + batch status
!-------------------------------------------------------------------------------
! alters ::    - this SeqNN fit to minimize loss on training data
!              - target_labels, [conv_input, dense_input] shuffled in place
!-------------------------------------------------------------------------------
subroutine snn_fit(this, batch_size, epochs, learn_rate, loss, &
                   conv_input, dense_input, verbose, &
                   target_labels, target_images)
    class(SeqNN)              :: this
    integer, intent(in)       :: batch_size, epochs
    real, intent(in)          :: learn_rate
    character(*), intent(in)  :: loss
    real(kind=8), optional    :: conv_input(:,:,:,:), dense_input(:,:), &
                                 target_labels(:,:), target_images(:,:,:,:)
    integer, optional         :: verbose
    real(kind=8), allocatable :: conv_x(:,:,:,:), dense_x(:,:), &
                                 labels(:,:), images(:,:,:,:)
    integer                   :: batches, input_i, i, j
    real(kind=8)              :: loss_val

    if (.not. this%is_init) then
        call this%snn_init(batch_size)
    end if

    if (associated(this%cnn)) then
        ! cnn present; must only pass conv_input
        if (.not. present(conv_input) .or. present(dense_input)) then
            print *, '-----------------------------------------'
            print *, '(sequential_neural_net :: snn_fit)'
            print *, 'ConvLayers present: only pass conv_input.'
            print *, '-----------------------------------------'
            stop -1
        end if

        ! whole batch count; truncating remainder skips last partial batch
        batches = size(conv_input, dim=4) / batch_size
    else
        ! cnn not present; must only pass dense_input
        if (.not. present(dense_input) .or. present(conv_input)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_fit)'
            print *, 'ConvLayers not present: only pass dense_input.'
            print *, '----------------------------------------------'
            stop -1
        end if

        ! whole batch count; truncating remainder skips last partial batch
        batches = size(dense_input, dim=1) / batch_size
    end if

    if (associated(this%dnn)) then
        ! dnn output, must pass target_labels
        if (.not. present(target_labels) .or. present(target_images)) then
            print *, '------------------------------------------'
            print *, '(sequential_neural_net :: snn_fit)'
            print *, 'DenseLayer output: only pass target_labels.'
            print *, '------------------------------------------'
            stop -1
        end if
    else
        ! cnn output, must pass target_images
        if (.not. present(target_images) .or. present(target_labels)) then
            print *, '------------------------------------------'
            print *, '(sequential_neural_net :: snn_fit)'
            print *, 'ConvLayer output: only pass target_images.'
            print *, '------------------------------------------'
            stop -1
        end if
    end if

    do i = 1, epochs
        if (present(verbose) .and. verbose > 0) then
            print *, 'epoch:', i
        end if
        
        input_i = 1 ! index of batch examples in input

        ! shuffle input and labels to same new ordering (improves training)
        if (associated(this%cnn)) then
            if (associated(this%dnn)) then
                ! cnn input, dnn output
                call pair_shuffle_4D_2D(conv_input, target_labels)
            else
                ! cnn input and output
                call pair_shuffle_4D_4D(conv_input, target_images)
            end if
        else
            ! dnn input and output
            call pair_shuffle_2D_2D(dense_input, target_labels)
        end if
        
        do j = 1, batches
            if (mod(j, 20) == 0) then
                if (present(verbose) .and. verbose > 1) then
                    print *, 'batch:', j, '/', batches
                end if
            end if

            ! extract corresponding batches of input and labels
            ! (slice the batch rows starting at input_i)
            if (associated(this%cnn)) then
                if (associated(this%dnn)) then
                    ! cnn input, dnn output
                    conv_x = conv_input(:,:,:,input_i:input_i+batch_size-1)
                    labels = target_labels(input_i:input_i+batch_size-1, :)

                    call this%snn_forw_prop(.true., conv_batch=conv_x)
                    call this%snn_back_prop(loss, target_labels=labels)
                    call this%snn_update(learn_rate, .true., conv_batch=conv_x)
                else
                    ! cnn input and output
                    conv_x = conv_input(:,:,:,input_i:input_i+batch_size-1)
                    images = target_images(:,:,:,input_i:input_i+batch_size-1)

                    call this%snn_forw_prop(.true., conv_batch=conv_x)
                    call this%snn_back_prop(loss, target_images=images)
                    call this%snn_update(learn_rate, .true., conv_batch=conv_x)
                end if
            else
                ! dnn input and output
                dense_x = dense_input(input_i:input_i+batch_size-1, :)
                labels = target_labels(input_i:input_i+batch_size-1, :)

                call this%snn_forw_prop(.true., dense_batch=dense_x)
                call this%snn_back_prop(loss, target_labels=labels)
                call this%snn_update(learn_rate, .true., dense_batch=dense_x)
            end if

            ! move index to start of next batch
            input_i = input_i + batch_size
        end do

        ! calculate loss
        if (present(verbose) .and. verbose > 0) then
            if (mod(i, 5) == 0) then

                ! do a forward pass for updated results, then calculate loss
                if (associated(this%cnn)) then
                    if (associated(this%dnn)) then
                        ! cnn input, dnn output
                        call this%snn_forw_prop(.false., conv_batch=conv_x)

                        loss_val = lossfunc_2D(this%dnn%output%a, labels, loss)
                    else
                        ! cnn input and output
                        call this%snn_forw_prop(.false., conv_batch=conv_x)

                        if (associated(this%cnn%output%next_pool)) then
                            ! pooling present
                            loss_val = lossfunc_4D( &
                                                this%cnn%output%next_pool%a, &
                                                images, loss)
                        else
                            ! no pooling
                            loss_val = lossfunc_4D( &
                                                this%cnn%output%a, images, loss)
                        end if
                    end if
                else
                    ! dnn input and output
                    call this%snn_forw_prop(.false., dense_batch=dense_x)

                    loss_val = lossfunc_2D(this%dnn%output%a, labels, loss)
                end if

                print *, '----------------------'
                print *, 'last train batch loss:'
                print *, loss_val
                print *, '----------------------'
            end if
        end if
    end do
end subroutine

!-------------------------------------------------------------------------------
! handles checking SeqNN loss on all given data (input, labels or images);
! only to be used with regression with real-valued labels
!
! must only pass conv_input if ConvLayers in SeqNN, otherwise
! must only pass dense_input if no ConvLayers present
!-------------------------------------------------------------------------------
! this:        (SeqNN - implicitly passed)
! loss:        (characters) loss function
!
! conv_input:    (optional - real(:,:,:,:)) input for ConvLayers
! dense_input:   (optional - real(:,:)) input for DenseLayers
! target_labels: (optional - real(:,:)) all targets we are trying to predict
! target_images: (optional - real(:,:,:,:)) all images we are trying to predict
! verbose:       (optional - integer) 0 = none, 2 = batch status
!-------------------------------------------------------------------------------
! returns:       (real) this SeqNN's loss on the given data
!-------------------------------------------------------------------------------
real(kind=8) function snn_regression_loss(this, loss, conv_input, dense_input, &
                                          target_labels, target_images, verbose)
    class(SeqNN)              :: this
    character(*), intent(in)  :: loss
    real(kind=8), optional    :: conv_input(:,:,:,:), dense_input(:,:), &
                                 target_labels(:,:), target_images(:,:,:,:)
    integer, optional         :: verbose
    real(kind=8), allocatable :: conv_x(:,:,:,:), dense_x(:,:), &
                                 labels(:,:), images(:,:,:,:)
    integer                   :: batch_size, batches, input_i, i
    real(kind=8)              :: total_loss, loss_val

    batch_size = this%batch_size

    if (associated(this%cnn)) then
        ! cnn present; must only pass conv_input
        if (.not. present(conv_input) .or. present(dense_input)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_regression_loss)'
            print *, 'ConvLayers present: only pass conv_input.'
            print *, '----------------------------------------------'
            stop -1
        end if

        ! whole batch count; truncating remainder skips last partial batch
        batches = size(conv_input, dim=4) / batch_size
    else
        ! cnn not present; must only pass dense_input
        if (.not. present(dense_input) .or. present(conv_input)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_regression_loss)'
            print *, 'ConvLayers not present: only pass dense_input.'
            print *, '----------------------------------------------'
            stop -1
        end if

        ! whole batch count; truncating remainder skips last partial batch
        batches = size(dense_input, dim=1) / batch_size
    end if

    if (associated(this%dnn)) then
        ! dnn output, must pass target_labels
        if (.not. present(target_labels) .or. present(target_images)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_regression_loss)'
            print *, 'DenseLayer output: only pass target_labels.'
            print *, '----------------------------------------------'
            stop -1
        end if
    else
        ! cnn output, must pass target_images
        if (.not. present(target_images) .or. present(target_labels)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_regression_loss)'
            print *, 'ConvLayer output: only pass target_images.'
            print *, '----------------------------------------------'
            stop -1
        end if
    end if

    total_loss = 0 ! keep total for later average
    input_i = 1    ! index of batch examples in input
    
    do i = 1, batches
        if (mod(i, 20) == 0) then
            if (present(verbose) .and. verbose > 1) then
                print *, 'batch:', i, '/', batches
            end if
        end if

        ! extract corresponding batches of input and labels
        ! (slice the batch rows starting at input_i)
        if (associated(this%cnn)) then
            if (associated(this%dnn)) then
                ! cnn input, dnn output
                conv_x = conv_input(:,:,:,input_i:input_i+batch_size-1)
                labels = target_labels(input_i:input_i+batch_size-1, :)

                call this%snn_forw_prop(.false., conv_batch=conv_x)
            else
                ! cnn input and output
                conv_x = conv_input(:,:,:,input_i:input_i+batch_size-1)
                images = target_images(:,:,:,input_i:input_i+batch_size-1)

                call this%snn_forw_prop(.false., conv_batch=conv_x)
            end if
        else
            ! dnn input and output
            dense_x = dense_input(input_i:input_i+batch_size-1, :)
            labels = target_labels(input_i:input_i+batch_size-1, :)

            call this%snn_forw_prop(.false., dense_batch=dense_x)
        end if

        if (associated(this%dnn)) then
            ! dnn output
            loss_val = lossfunc_2D(this%dnn%output%a, labels, loss)
        else
            ! cnn output
            if (associated(this%cnn%output%next_pool)) then
                ! pooling present
                loss_val = lossfunc_4D(this%cnn%output%next_pool%a, &
                                       images, loss)
            else
                ! no pooling
                loss_val = lossfunc_4D(this%cnn%output%a, images, loss)
            end if
        end if

        total_loss = total_loss + loss_val

        ! move index to start of next batch
        input_i = input_i + batch_size
    end do

    snn_regression_loss = total_loss / batches ! avg loss
end function

!-------------------------------------------------------------------------------
! handles checking SeqNN accuracy on all given data (input, labels);
! only to be used with classification with one-hot encoded labels;
! output must be DenseLayer;
!
! must only pass conv_input if ConvLayers in SeqNN, otherwise
! must only pass dense_input if no ConvLayers present
!-------------------------------------------------------------------------------
! this:          (SeqNN - implicitly passed)
! target_labels: (real(:,:)) all ONE-HOT ENCODED targets to predict
!                   *** see net_helper_procedures :: one_hot_encode_special
!
! conv_input:    (optional - real(:,:,:,:)) input for ConvLayers
! dense_input:   (optional - real(:,:)) input for DenseLayers
! verbose:       (optional - integer) 0 = none, 2 = batch status
!-------------------------------------------------------------------------------
! returns:       (real) this SeqNN's one-hot accuracy on the given data
!-------------------------------------------------------------------------------
real(kind=8) function snn_one_hot_accuracy(this, target_labels, &
                                           conv_input, dense_input, verbose)
    class(SeqNN)                       :: this
    real(kind=8), intent(in)           :: target_labels(:,:)
    real(kind=8), intent(in), optional :: conv_input(:,:,:,:), dense_input(:,:)
    integer, optional                  :: verbose
    real(kind=8), allocatable          :: conv_x(:,:,:,:), dense_x(:,:), &
                                          labels(:,:)
    integer                            :: batch_size, batches, input_i, i
    real(kind=8)                       :: total_correct_prob

    batch_size = this%batch_size

    if (associated(this%cnn)) then
        ! cnn present; must only pass conv_input
        if (.not. present(conv_input) .or. present(dense_input)) then
            print *, '----------------------------------------------------'
            print *, '(sequential_neural_net :: snn_test_one_hot_accuracy)'
            print *, 'ConvLayers present: only pass conv_input.'
            print *, '----------------------------------------------------'
            stop -1
        end if

        ! whole batch count; truncating remainder skips last partial batch
        batches = size(conv_input, dim=4) / batch_size
    else
        ! cnn not present; must only pass dense_input
        if (.not. present(dense_input) .or. present(conv_input)) then
            print *, '----------------------------------------------------'
            print *, '(sequential_neural_net :: snn_test_one_hot_accuracy)'
            print *, 'ConvLayers not present: only pass dense_input.'
            print *, '----------------------------------------------------'
            stop -1
        end if

        ! whole batch count; truncating remainder skips last partial batch
        batches = size(dense_input, dim=1) / batch_size
    end if

    total_correct_prob = 0 ! keep total for later average
    input_i = 1            ! index of batch examples in input

    do i = 1, batches
        if (mod(i, 20) == 0) then
            if (present(verbose) .and. verbose > 1) then
                print *, 'batch:', i, '/', batches
            end if
        end if

        ! extract corresponding batches of input and labels
        ! (slice the batch rows starting at input_i)
        if (associated(this%cnn)) then
            conv_x = conv_input(:,:,:,input_i:input_i+batch_size-1)
            labels = target_labels(input_i : input_i+batch_size-1, :)
            call this%snn_forw_prop(.false., conv_batch=conv_x)
        else
            dense_x = dense_input(input_i:input_i+batch_size-1, :)
            labels = target_labels(input_i : input_i+batch_size-1, :)
            call this%snn_forw_prop(.false., dense_batch=dense_x)
        end if

        ! dnn%output%a has prediction vector upon completion
        total_correct_prob = total_correct_prob + &
                             one_hot_accuracy_2D(this%dnn%output%a, labels)

        input_i = input_i + batch_size
    end do

    snn_one_hot_accuracy = total_correct_prob / batches ! avg probability
end function

!-------------------------------------------------------------------------------
! handles predicting with trained SeqNN on all given input data
!
! must only pass conv_input if ConvLayers in SeqNN, otherwise
! must only pass dense_input if no ConvLayers present
!-------------------------------------------------------------------------------
! this:         (SeqNN - implicitly passed)
! res:          (:,:) stores predictions; prediction rows correspond to input
!
! conv_input:   (optional - real(:,:,:,:)) input for ConvLayers
! dense_input:  (optional - real(:,:)) input for DenseLayers
!-------------------------------------------------------------------------------
! alters ::     res becomes predictions of this SeqNN on input data
!-------------------------------------------------------------------------------
subroutine snn_predict(this, res, conv_input, dense_input)
    class(SeqNN)                       :: this
    real(kind=8), allocatable          :: res(:,:), conv_x(:,:,:,:), &
                                          dense_x(:,:)
    real(kind=8), intent(in), optional :: conv_input(:,:,:,:), dense_input(:,:)
    integer                            :: batch_size, items, batches, input_i, &
                                          i, remain

    batch_size = this%batch_size

    if (associated(this%cnn)) then
        ! cnn present; must only pass conv_input
        if (.not. present(conv_input) .or. present(dense_input)) then
            print *, '-----------------------------------------'
            print *, '(sequential_neural_net :: snn_fit)'
            print *, 'ConvLayers present: only pass conv_input.'
            print *, '-----------------------------------------'
            stop -1
        end if

        items = size(conv_input, dim=4)
    else
        ! cnn not present; must only pass dense_input
        if (.not. present(dense_input) .or. present(conv_input)) then
            print *, '----------------------------------------------'
            print *, '(sequential_neural_net :: snn_fit)'
            print *, 'ConvLayers not present: only pass dense_input.'
            print *, '----------------------------------------------'
            stop -1
        end if

        items = size(dense_input, dim=1)
    end if

    ! whole batch count; truncating remainder skips last partial batch
    batches = items / batch_size
    input_i = 1 ! index of batch examples in input

    ! allocate a prediction row for each row in input
    if (allocated(res)) then
        if (.not. all(shape(res) == [items, this%dnn%output%out_nodes])) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(items, this%dnn%output%out_nodes))
    end if

    do i = 1, batches
        ! extract whole input batch (slice the batch items starting at i) 
        if (associated(this%cnn)) then
            conv_x = conv_input(:,:,:,(i-1)*batch_size+1:i*batch_size)
            call this%snn_forw_prop(.false., conv_batch=conv_x)
        else
            dense_x = dense_input((i-1)*batch_size+1:i*batch_size,:)
            call this%snn_forw_prop(.false., dense_batch=dense_x)
        end if

        ! record predictions
        res(input_i:input_i+batch_size-1, :) = this%dnn%output%a
        input_i = input_i + batch_size
    end do

    ! predict for remaining inputs that were truncated above
    remain = items - batches * batch_size

    if (remain > 0) then
        if (associated(this%cnn)) then

            ! handle if batch wasn't already made
            if (.not. allocated(conv_x)) then
                allocate(conv_x(this%cnn%in_dims(1), &
                                this%cnn%in_dims(2), &
                                this%cnn%in_dims(3), &
                                batch_size))
            end if

            conv_x = 0

            ! batch currently allocated to proper batch shape;
            ! overwrite the items we need, ignore remainder
            conv_x(:,:,:,:remain) = conv_input(:,:,:,items-remain+1:)
            call this%snn_forw_prop(.false., conv_batch=conv_x)
        else
            ! handle if batch wasn't already made
            if (.not. allocated(dense_x)) then
                allocate(dense_x(batch_size, this%dnn%in_nodes))
            end if

            dense_x = 0

            ! batch currently allocated to proper batch shape;
            ! overwrite the items we need, ignore remainder
            dense_x(:remain, :) = dense_input(items-remain+1:,:)
            call this%snn_forw_prop(.false., dense_batch=dense_x)
        end if

        ! fill last section of predictions with remaining items
        res(input_i:, :) = this%dnn%output%a(:remain, :)
    end if
end subroutine

!-------------------------------------------------------------------------------
! prints the dimensions output by each layer in this SeqNN
!-------------------------------------------------------------------------------
! this:     (SeqNN - implicitly passed)
!-------------------------------------------------------------------------------
! alters :: prints details to stdout
!-------------------------------------------------------------------------------
subroutine snn_summary(this)
    class(SeqNN)               :: this
    class(ConvLayer), pointer  :: curr_conv
    class(DenseLayer), pointer :: curr_dense

    print *, '----------------------'

    ! loop through ConvNN's ConvLayers (and PoolLayers)
    if (associated(this%cnn)) then
        print *, 'dimensions:               rows        cols    channels'
        print *, '-----------'

        print *, 'ConvLayer input:  ', this%cnn%in_dims

        curr_conv => this%cnn%first_hid

        do while (associated(curr_conv))
            print *, 'ConvLayer output: ', &
                curr_conv%out_rows, curr_conv%out_cols, curr_conv%out_channels

            if (associated(curr_conv%next_pool)) then
                print *, 'PoolLayer output: ', &
                    curr_conv%next_pool%out_rows, &
                    curr_conv%next_pool%out_cols, &
                    curr_conv%next_pool%out_channels
            end if

            curr_conv => curr_conv%next_layer
        end do
        print *, '-----------'
    end if

    ! loop through DenseNN's DenseLayers
    if (associated(this%dnn)) then
        print *, 'dimensions:              nodes'
        print *, '-----------'

        ! loop through DenseNN's DenseLayers
        print *, 'DenseLayer input: ', this%dnn%in_nodes

        curr_dense => this%dnn%first_hid

        do while (associated(curr_dense))
            print *, 'DenseLayer output:', curr_dense%out_nodes

            curr_dense => curr_dense%next_layer
        end do
    end if
    print *, '----------------------'
end subroutine
end module
