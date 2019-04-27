!-------------------------------------------------------------------------------
! implementation for a dense, fully-connected layer type (DenseLayer)
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module dense_layer_definitions
use net_helper_procedures
implicit none

!===============================================================================
!===============================================================================
! procedures with 2D array input require variables-as-columns form
!===============================================================================
!===============================================================================

!===============================================================================
! types
!===============================================================================

! represents a Dense (fully-connected) Layer in a neural network
!
! we define a DenseLayer as group of nodes and the weights that feed into them
! (which is difsferent than the usual idea of a dense layer just being nodes).
!
! consider a simple neural network with weights w1 and w2,
! and nodes n1 (input), n2 (hidden), and n3 (output):
! 
!                    O ----  O  ---- O
!
!                   n1  w1  n2   w2  n3
!
! in the definition used here, there would be only two DenseLayers:
! the first hidden DenseLayer (w1,n2), and the output DenseLayer (w2,n3).
! node n1 is not a DenseLayer on its own, becuase no weights feed into it
!
! in this implementation, all examples in a batch are processed in the same
! forward pass by utilizing more extensive matrix operations
!
! DenseLayer are connected to each other as a doubly-linked list
type :: DenseLayer
    integer                    :: in_nodes, out_nodes, batch_size
    class(DenseLayer), pointer :: prev_layer, next_layer
    character(len=20)          :: activ
    real, allocatable          :: w(:,:), & ! weights
                                  b(:,:), & ! biases
                                  z(:,:), & ! weighted inputs
                                  a(:,:), & ! activations
                                  d(:,:), & ! deltas (errors),
                                  drop(:,:) ! dropout inputs (0=drop, 1=keep)
    real                       :: drop_rate ! % of input nodes to drop
contains
    ! procedures that traverse through linked list of DenseLayers
    procedure, pass            :: dense_init, dense_update, &
                                  dense_forw_prop, dense_back_prop, &
                                  dense_dropout_rand
end type
contains

!===============================================================================
! constructors / destructors
!===============================================================================

!-------------------------------------------------------------------------------
! constructs a new DenseLayer
!-------------------------------------------------------------------------------
! in_nodes:  (integer) nodes feeding into DenseLayer
! out_nodes: (integer) nodes output by this DenseLayer
! activ:     (characters) activation function
! drop_rate: (real) % of input nodes to dropout
!-------------------------------------------------------------------------------
! returns :: (DenseLayer pointer) new DenseLayer
!-------------------------------------------------------------------------------
function create_dense_layer(in_nodes, out_nodes, activ, drop_rate)
    class(DenseLayer), pointer :: create_dense_layer
    integer, intent(in)        :: in_nodes, out_nodes
    character(*), intent(in)   :: activ
    real, intent(in)           :: drop_rate

    allocate(create_dense_layer)
    create_dense_layer%in_nodes   =  in_nodes
    create_dense_layer%out_nodes  =  out_nodes
    create_dense_layer%activ      =  activ
    create_dense_layer%drop_rate  =  drop_rate
    create_dense_layer%prev_layer => null()
    create_dense_layer%next_layer => null()
end function

!-------------------------------------------------------------------------------
! deallocate a DenseLayer
!-------------------------------------------------------------------------------
! l:        (DenseLayer pointer)
!-------------------------------------------------------------------------------
! alters :: l is deallocated
!-------------------------------------------------------------------------------
subroutine deallocate_dense_layer(l)
    class(DenseLayer), pointer :: l
    deallocate(l%w, l%b, l%z, l%a, l%d)
    deallocate(l)
end subroutine

!===============================================================================
! DenseLayer procedures
!===============================================================================

!-------------------------------------------------------------------------------
! helper subroutine to initialize the various matrices of DenseLayers
!-------------------------------------------------------------------------------
! this:       (DenseLayer - implicitly passed)
! batch_size: (integer) examples to process (at once) before back prop
!-------------------------------------------------------------------------------
! alters ::   this DenseLayer's matrices are allocated and become usable
!-------------------------------------------------------------------------------
subroutine dense_init(this, batch_size)
    class(DenseLayer)   :: this
    integer, intent(in) :: batch_size

    this%batch_size = batch_size

    allocate(this%w(this%in_nodes, this%out_nodes), &
             this%b(batch_size, this%out_nodes), &
             this%z(batch_size, this%out_nodes), &
             this%a(batch_size, this%out_nodes), &
             this%d(batch_size, this%out_nodes), &
             this%drop(batch_size, this%in_nodes))

    ! initialize weights and biases
    call random_number(this%w)   ! random weights, uniform range: [0,1)
    this%w = (this%w - 0.5) / 10 ! shift and scale to [-0.05, 0.05)
    this%b = 0

    ! traverse next layers
    if (associated(this%next_layer)) then
        call this%next_layer%dense_init(batch_size)
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to random initialize/overwrite a DenseLayers dropout array
!-------------------------------------------------------------------------------
! this:       (DenseLayer - implicitly passed)
!-------------------------------------------------------------------------------
! alters ::   this DenseLayer's drop array has randomized values
!-------------------------------------------------------------------------------
subroutine dense_dropout_rand(this)
    class(DenseLayer)   :: this

    ! initialize dropout layer; 0=drop, 1=keep (we will be multiplying)
    if (this%drop_rate > 0) then
        call random_number(this%drop) ! random weights, uniform range: [0,1)

        ! this%drop_rate are dropped, so 1 - this%drop_rate are kept;
        ! think of keep = 1 - this%drop_rate; add keep, then cast to int to
        ! truncate kept values to 1 and dropped values to 0;
        !
        ! shift to [keep, 1+keep), so after truncating, so
        ! values in top keep proportion are kept at 1
        this%drop = int(this%drop + 1 - this%drop_rate)
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to forward propagate through DenseLayers
!-------------------------------------------------------------------------------
! this:     (DenseLayer - implicitly passed)
! input:    (real(:,:)) input batch to forward propagate
! is_train: (logical) in training iteration
!-------------------------------------------------------------------------------
! alters :: this DenseLayer's z and a are calculated
!-------------------------------------------------------------------------------
subroutine dense_forw_prop(this, input, is_train)
    class(DenseLayer)   :: this
    real, intent(in)    :: input(:,:)
    logical, intent(in) :: is_train

    ! z(l) = matmul(a(l-1), w(l)) + b(l); first handle dropout
    if (this%drop_rate > 0 .and. is_train) then
        call this%dense_dropout_rand() ! randomize dropout
        this%z = matmul(input*this%drop, this%w) + this%b
    else
        this%z = matmul(input, this%w) + this%b ! no drops
    end if

    ! apply activation and traverse next layers (if this is not output layer);
    ! output layer has different activation function usage
    if (associated(this%next_layer)) then
        if (this%activ == 'softmax') then
            print *, '--------------------------------------------'
            print *, '(dense_layer_definitions :: dense_forw_prop)'
            print *, 'invalid activation function.'
            print *, 'softmax only supported for output layer.'
            print *, 'supported: sigmoid, relu, leaky_relu, elu'
            print *, '--------------------------------------------'
            stop -1
        end if

        ! a(l) = activ(z(l))
        this%a = activfunc(this%z, this%activ)
        call this%next_layer%dense_forw_prop(this%a, is_train)
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to back propagate through DenseLayers
!-------------------------------------------------------------------------------
! this:     (DenseLayer - implicitly passed)
!-------------------------------------------------------------------------------
! alters :: this DenseLayer's d is calculated
!-------------------------------------------------------------------------------
subroutine dense_back_prop(this)
    class(DenseLayer) :: this

    if (.not. associated(this%next_layer)) then
        print *, '--------------------------------------------'
        print *, '(dense_layer_definitions :: dense_back_prop)'
        print *, 'cannot call dense_back_prop on output layer.'
        print *, '--------------------------------------------'
        stop -1
    end if

    ! deltas for this layer in terms of next layer deltas:
    ! delta(l) = matmul(delta(l+1), transpose(w(l+1))) * activ_deriv(z(l))
    this%d = matmul(this%next_layer%d, transpose(this%next_layer%w)) * &
             activfunc_deriv(this%z, this%activ)

    ! traverse previous layers
    if (associated(this%prev_layer)) then
        call this%prev_layer%dense_back_prop()
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to adjust weights and biases in DenseLayers
!-------------------------------------------------------------------------------
! this:       (DenseLayer - implicitly passed)
! input:      (real(:,:)) previous layer activations
! learn_rate: (real) scale factor for change in weights and biases
! is_train:   (logical) in training iteration
!-------------------------------------------------------------------------------
! alters ::   this DenseLayer's weights and biases adjusted to minimize loss
!-------------------------------------------------------------------------------
subroutine dense_update(this, input, learn_rate, is_train)
    class(DenseLayer)   :: this
    real, intent(in)    :: input(:,:), learn_rate
    logical, intent(in) :: is_train
    real, allocatable   :: avg_change_row(:)
    real                :: scale
    integer             :: r

    ! multiply by learn_rate, then average across all examples in batch
    scale = learn_rate / this%batch_size

    ! weights:
    ! w = w - avg_change (also handling dropout)
    ! avg_change = matmul(transpose(a(l-1)), delta(l)) * scale
    if (this%drop_rate > 0 .and. is_train) then
        call this%dense_dropout_rand() ! randomize dropout
        this%w = this%w - matmul(transpose(input*this%drop), this%d) * scale
    else
        this%w = this%w - matmul(transpose(input), this%d) * scale ! no drops
    end if

    ! biases:
    ! each row in the deltas corresponds to one example
    ! in the batch. we want to update each example's biases
    ! by the same amount, so we find the average row
    ! in the batch (the vector of col-averages across the batch)
    !
    ! b = (each row of b) - avg_change_row
    ! avg_change_row = col_sums(delta(l)) * scale
    avg_change_row = sum(this%d, dim=1) * scale ! dim=1 finds col sums

    ! update each row of b
    do r = 1, this%batch_size
        this%b(r,:) = this%b(r,:) - avg_change_row
    end do

    ! traverse next layers
    if (associated(this%next_layer)) then
        call this%next_layer%dense_update(this%a, learn_rate, is_train)
    end if
end subroutine
end module
