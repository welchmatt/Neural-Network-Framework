!-------------------------------------------------------------------------------
! implementation for DenseLayer (fully-connected) type (see below)
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module dense_layer_definitions
use net_helper_procedures
implicit none

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
                                  d(:,:)    ! deltas (errors)
contains
    ! procedures that traverse through linked list of DenseLayers
    procedure, pass            :: dense_init, dense_update, &
                                  dense_forw_prop, dense_back_prop
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
!-------------------------------------------------------------------------------
! returns :: (DenseLayer pointer) new DenseLayer
!-------------------------------------------------------------------------------
function create_dense_layer(in_nodes, out_nodes, activ)
    class(DenseLayer), pointer :: create_dense_layer
    integer, intent(in)        :: in_nodes, out_nodes
    character(*), intent(in)   :: activ

    allocate(create_dense_layer)
    create_dense_layer%in_nodes   =  in_nodes
    create_dense_layer%out_nodes  =  out_nodes
    create_dense_layer%activ      =  activ
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
             this%d(batch_size, this%out_nodes))

    ! initialize weights and biases
    call random_number(this%w)   ! random weights, uniform range: [0,1)
    this%w = (this%w - 0.5) / 10 ! shift and scale to [-0.05, 0.05]
    this%b = 0

    ! traverse next layers
    if (associated(this%next_layer)) then
        call this%next_layer%dense_init(batch_size)
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to forward propagate through DenseLayers
!-------------------------------------------------------------------------------
! this:     (DenseLayer - implicitly passed)
! input:    (real(:,:)) input batch to forward propagate
!-------------------------------------------------------------------------------
! alters :: this DenseLayer's z and a are calculated
!-------------------------------------------------------------------------------
subroutine dense_forw_prop(this, input)
    class(DenseLayer) :: this
    real, intent(in)  :: input(:,:)

    ! z(l) = matmul(a(l-1), w(l)) + b(l)
    this%z = matmul(input, this%w) + this%b

    ! apply activation and traverse next layers (if this is not output layer);
    ! output layer has different activation function usage
    if (associated(this%next_layer)) then
        ! a(l) = activ(z(l))
        this%a = activfunc(this%z, this%activ)
        call this%next_layer%dense_forw_prop(this%a)
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
!-------------------------------------------------------------------------------
! alters ::   this DenseLayer's weights and biases adjusted to minimize loss
!-------------------------------------------------------------------------------
subroutine dense_update(this, input, learn_rate)
    class(DenseLayer) :: this
    real, intent(in)  :: input(:,:), learn_rate
    real, allocatable :: avg_change_row(:)
    real              :: scale
    integer           :: r

    ! multiply by learn_rate, then average across all examples in batch
    scale = learn_rate / this%batch_size

    ! weights:
    ! w = w - avg_change
    ! avg_change = matmul(transpose(a(l-1)), delta(l)) * scale
    this%w = this%w - matmul(transpose(input), this%d) * scale

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
        call this%next_layer%dense_update(this%a, learn_rate)
    end if
end subroutine
end module