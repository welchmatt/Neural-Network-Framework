!-------------------------------------------------------------------------------
! neural network implementation that utilizes ConvLayers and PoolLayers
! (from conv_layer_definitions.f08 and pool_layer_definitions.f08)
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module conv_neural_net
use net_helper_procedures
use conv_layer_definitions
use pool_layer_definitions
implicit none

!===============================================================================
!===============================================================================
! procedures with 4D array input require (rows, columns, channels, batches) form
!===============================================================================
!===============================================================================

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
!   *** require input and targets as real(batch, rows, columns, channels) arrays
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
end subroutine

!-------------------------------------------------------------------------------
! calculates delta for ConvNN's output layer (derivative of loss w.r.t. z)
!-------------------------------------------------------------------------------
! this:     (ConvNN - implicitly passed)
! targets:  (real(:,:,:,:)) targets we are trying to predict
! loss:     (characters) loss function
!-------------------------------------------------------------------------------
! alters :: program crashes if this ConvNN's output layer's d is not calculated
!-------------------------------------------------------------------------------
subroutine cnn_out_delta(this, targets, loss)
    class(ConvNN)            :: this
    real, intent(in)         :: targets(:,:,:,:)
    character(*), intent(in) :: loss

    select case (loss)
        case ('mse')
            select case(this%output%activ)
                case ('sigmoid', 'relu', 'leaky_relu', 'elu')
                    ! d(L) = (a(L) - targets) * out_activ_deriv(z(L));
                    ! (these are implemented with element-wise derivatives)
                    this%output%d = &
                            (this%output%a - targets) * &
                            activfunc_deriv(this%output%z, this%output%activ)

                case default
                    print *, '-----------------------------------------'
                    print *, '(conv_neural_net :: cnn_out_delta)'
                    print *, 'mse - invalid output activation function.'
                    print *, 'supported: sigmoid, relu, leaky_relu, elu'
                    print *, '-----------------------------------------'
                    stop -1
                end select

        case default
            print *, '----------------------------------'
            print *, '(conv_neural_net :: cnn_out_delta)'
            print *, 'invalid loss function.'
            print *, 'supported: mse'
            print *, '----------------------------------'
            stop -1
    end select
end subroutine

!-------------------------------------------------------------------------------
! wrapper subroutine to back propagate through ConvNN's ConvLayers;
!-------------------------------------------------------------------------------
! this:           (ConvNN - implicitly passed)
! out_delta_done: (logical) true if output deltas calculated by other source
!
! targets:        (optional - real(:,:,:,:)) targets we are trying to predict
! loss:           (optional - characters) loss function
!-------------------------------------------------------------------------------
! alters ::       this ConvNN's ConvLayers' d's calculated
!-------------------------------------------------------------------------------
subroutine cnn_back_prop(this, out_delta_done, targets, loss)
    class(ConvNN)                      :: this
    logical, intent(in)                :: out_delta_done
    real, intent(in), optional         :: targets(:,:,:,:)
    character(*), intent(in), optional :: loss

    ! if out deltas not calculated by other source, we must calculate them
    if (.not. out_delta_done) then
        ! require targets and loss functino to calculate out deltas
        if (.not. present(targets) .or. .not. present(loss)) then
            print *, '------------------------------------------------'
            print *, '(conv_neural_net :: cnn_back_prop)'
            print *, 'must pass targets, loss to calculate out deltas.'
            print *, '------------------------------------------------'
            stop -1
        end if
        
        call this%cnn_out_delta(targets, loss)
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
end subroutine
end module
