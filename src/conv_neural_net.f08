!-------------------------------------------------------------------------------
! neural network implementation that utilizes ConvLayers (convolutional)
!
! *** for now, this can only be used feeding into a DenseNN with DenseLayers,
!     whereby the derivative wrt this ConvNN's output layer is calculated during
!     backpropagation of the DenseLayers
! *** this cannot be utilized as a network on its own, because calcualting the
!     derivative of cost wrt output layer based on some labels is not
!     implemented (which would only be used for a Fully Convolutional Network,
!     whereby the output kernels are directly compared to labels, and not
!     propagated directly to DenseLayers)
! *** implementing this feature simply requires adjusting the cnn_out_delta
!     function to accept labels and a loss function, and to calculate the proper
!     derivative; for now, this feature is not needed, because I am not yet
!     trying to model a Fully Convolutional Network
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module conv_neural_net
use net_helper_procedures
use conv_layer_definitions
use pool_layer_definitions
implicit none

!===============================================================================
! types
!===============================================================================

! represents a neural network with ConvLayers
!
! this type serves as a wrapper for a doubly linked list of ConvLayers,
! with each layer possibly followed by a PoolLayer
type :: ConvNN
    ! head (first_hid), tail (output) of wrapped linked list
    class(ConvLayer), pointer :: first_hid, output
    integer                   :: in_dims(3), out_rows, out_cols, out_channels, &
                                 out_count, batch_size
    logical                   :: is_init
contains
    ! wrapper procedures that initiate ConvLayer traversals
    procedure, pass           :: cnn_add_layer, cnn_add_pool_layer, cnn_init, &
                                 cnn_forw_prop, cnn_out_delta, cnn_back_prop, &
                                 cnn_update
end type
contains

!===============================================================================
! constructors / destructors
!===============================================================================

!-------------------------------------------------------------------------------
! constructs a new ConvNN
!-------------------------------------------------------------------------------
! input_dims: (integer(3)) (height, width, channels) of one input
!-------------------------------------------------------------------------------
! returns ::  (ConvNN pointer) new ConvNN
!-------------------------------------------------------------------------------
function create_cnn(input_dims)
    class(ConvNN), pointer :: create_cnn
    integer, intent(in)    :: input_dims(3)

    allocate(create_cnn)
    create_cnn%in_dims   =  input_dims
    create_cnn%is_init   = .false.
    create_cnn%first_hid => null()
    create_cnn%output    => null()
end function

!-------------------------------------------------------------------------------
! deallocate a ConvNN
!-------------------------------------------------------------------------------
! cnn:      (ConvNN pointer)
!-------------------------------------------------------------------------------
! alters :: cnn and the ConvLayers (and PoolLayers) it wraps are deallocated
!-------------------------------------------------------------------------------
subroutine deallocate_cnn(cnn)
    class(ConvNN), pointer    :: cnn
    class(ConvLayer), pointer :: curr_layer

    if (cnn%is_init) then
        curr_layer => cnn%first_hid%next_layer ! second layer

        ! traverse layers, deallocating previous layer
        do while(associated(curr_layer))
            call deallocate_conv_layer(curr_layer%prev_layer)
            curr_layer => curr_layer%next_layer
        end do

        ! output layer not a 'previous' layer; manually deallocate it
        call deallocate_conv_layer(cnn%output)
        deallocate(cnn)
    end if
end subroutine

!===============================================================================
! ConvNN procedures
!   *** all require input and labels in variables-as-columns form
!===============================================================================

!-------------------------------------------------------------------------------
! create and add a new ConvLayer to the tail of this ConvNN's linked list
!-------------------------------------------------------------------------------
! this:        (ConvNN - implicitly passed)
! kernels:     (integer) kernels in new layer
! kernel_dims: (integer(2)) (height, width) of each kernel channel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! activ:       (characters) activation function
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! alters ::   new ConvLayer appended to this ConvNN's linked list
!-------------------------------------------------------------------------------
subroutine cnn_add_layer(this, kernels, kernel_dims, stride, activ, padding)
    class(ConvNN)             :: this
    integer, intent(in)       :: kernels, kernel_dims(2), stride(2)
    character(*), intent(in)  :: activ, padding
    class(ConvLayer), pointer :: new_layer
    integer                   :: input_dims(3)

    ! make new tail matrices align with current tail matrices
    if (associated(this%output)) then
        ! new tail fed by existing tail
        if (associated(this%output%next_pool)) then
            ! fed by Pool of tail
            input_dims = [this%output%next_pool%out_rows, &
                          this%output%next_pool%out_cols, &
                          this%output%next_pool%out_channels]
        else
            ! fed by tail directly
            input_dims = [this%output%out_rows, &
                          this%output%out_cols, &
                          this%output%out_channels]
        end if
    else
        ! first layer fed by input variables
        input_dims = this%in_dims
    end if

    new_layer => create_conv_layer(input_dims, kernels, kernel_dims, stride, &
                                   activ, padding)

    if (associated(this%output)) then
        ! new tail appended to existing tail
        this%output%next_layer => new_layer
        new_layer%prev_layer => this%output
    else
        ! adding first layer
        this%first_hid => new_layer
    end if

    this%output => new_layer

    ! outward facing node counts; easier interfacing with overall output
    this%out_rows     = new_layer%out_rows
    this%out_cols     = new_layer%out_cols
    this%out_channels = new_layer%out_channels
    this%out_count    = new_layer%out_count
end subroutine

!-------------------------------------------------------------------------------
! create and add a new PoolLayer to tail ConvLayer of this ConvNN's linked list
!-------------------------------------------------------------------------------
! kernel_dims: (integer(2)) (height, width) of pool kernel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! pool:        (characters) pool type
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! alters ::    new PoolLayer added to this ConvNN's tail ConvLayer
!-------------------------------------------------------------------------------
subroutine cnn_add_pool_layer(this, kernel_dims, stride, pool, padding)
    class(ConvNN)            :: this
    integer, intent(in)      :: kernel_dims(2), stride(2)
    character(*), intent(in) :: pool, padding

    if (associated(this%output)) then
        call  this%output%conv_add_pool_layer(kernel_dims, stride, pool, &
                                              padding)
    else
        print *, '---------------------------------------'
        print *, '(conv_neural_net :: cnn_add_pool_layer)'
        print *, 'PoolLayer must follow ConvLayer.'
        print *, '---------------------------------------'
        stop -1
    end if

    ! outward facing node counts now from last PoolLayer
    this%out_rows     = this%output%next_pool%out_rows
    this%out_cols     = this%output%next_pool%out_cols
    this%out_channels = this%output%next_pool%out_channels
    this%out_count    = this%output%next_pool%out_count
end subroutine

!-------------------------------------------------------------------------------
! initialize all ConvLayers in this ConvNN, and specify batch size to process
!-------------------------------------------------------------------------------
! this:       (ConvNN - implicitly passed)
! batch_size: (integer) examples to process before back prop
!-------------------------------------------------------------------------------
! alters ::   this ConvNN's ConvLayers are allocated and become usable
!-------------------------------------------------------------------------------
subroutine cnn_init(this, batch_size)
    class(ConvNN) :: this
    integer       :: batch_size

    this%is_init = .true.
    this%batch_size = batch_size
    call this%first_hid%conv_init(batch_size)

    this%output%d = 0 ! for cnn_out_delta check: see below
end subroutine

!-------------------------------------------------------------------------------
! wrapper subroutine to forward propagate input through ConvNN's ConvLayers
!-------------------------------------------------------------------------------
! this:     (ConvNN - implicitly passed)
! input:    (real(:,:,:,:)) input batch to forward propagate
!-------------------------------------------------------------------------------
! alters :: this ConvNN's ConvLayers' z's and a's calculated
!-------------------------------------------------------------------------------
subroutine cnn_forw_prop(this, input)
    class(ConvNN)    :: this
    real, intent(in) :: input(:,:,:,:)

    call this%first_hid%conv_forw_prop(input)

    ! different activation function on output layer
    ! a(l) = out_activ(z(l))
    call out_activfunc(this%output%z, this%output%activ, this%output%a)
end subroutine

!-------------------------------------------------------------------------------
! SEE FILE HEADER FOR DETAILS:
! crashes if the delta for ConvNN has not been set, otherwise does nothing:
!
! for now, we require DenseLayers after ConvLayers, therefore the
! output delta for the last ConvLayer is currently calculated by the DenseLayers
!
! cnn_init and and cnn_update sets this%output%d = 0, so it is clear to see if
! the output delta has been set by another source
!-------------------------------------------------------------------------------
! this:     (ConvNN - implicitly passed)
!-------------------------------------------------------------------------------
! alters :: program crashes if this ConvNN's output layer's d is not calculated
!-------------------------------------------------------------------------------
subroutine cnn_out_delta(this)
    class(ConvNN) :: this

    ! d explicitly set to 0 at start, so floating point comparison is exact
    if (all(this%output%d == 0)) then
        print *, '--------------------------------------'
        print *, '(conv_neural_net :: cnn_out_delta)'
        print *, 'output delta of ConvNN not calculated.'
        print *, 'must add DenseLayers after.'
        print *, '--------------------------------------'
        stop -1
    end if
end subroutine

!-------------------------------------------------------------------------------
! wrapper subroutine to back propagate through ConvNN's ConvLayers;
!-------------------------------------------------------------------------------
! this:     (ConvNN - implicitly passed)
!-------------------------------------------------------------------------------
! alters :: this ConvNN's ConvLayers' d's calculated
!-------------------------------------------------------------------------------
subroutine cnn_back_prop(this, out_delta_done)
    class(ConvNN)       :: this
    logical, intent(in) :: out_delta_done

    ! ensure out deltas are calculated
    if (.not. out_delta_done) then
        call this%cnn_out_delta()
    end if

    if (associated(this%output%prev_layer)) then
        call this%output%prev_layer%conv_back_prop()
    end if
end subroutine

!-------------------------------------------------------------------------------
! wrapper subroutine to adjust kernels and biases in ConvNN's ConvLayers
!-------------------------------------------------------------------------------
! this:       (ConvNN - implicitly passed)
! input:      (real(:,:,:,:)) input batch
! learn_rate: (real) scale factor for change in kernels and biases
!-------------------------------------------------------------------------------
! alters ::   this ConvNN's kernels and biases adjusted to minimize loss
!-------------------------------------------------------------------------------
subroutine cnn_update(this, input, learn_rate)
    class(ConvNN)     :: this
    real, intent(in)  :: input(:,:,:,:), learn_rate

    ! first hid a(l-1) is input batch
    call this%first_hid%conv_update(input, learn_rate)

    ! reset for next check; see cnn_out_delta and file header for details
    this%output%d = 0
end subroutine
end module