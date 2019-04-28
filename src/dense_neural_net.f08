!-------------------------------------------------------------------------------
! neural network implementation that utilizes DenseLayers
! (from dense_layer_definitions.f08)
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module dense_neural_net
use net_helper_procedures
use dense_layer_definitions
implicit none

!===============================================================================
!===============================================================================
! procedures with 2D array input require variables-as-columns form
!===============================================================================
!===============================================================================

!===============================================================================
! types
!===============================================================================

! represents a neural network with DenseLayers
!
! this type serves as a wrapper for a doubly linked list of DenseLayers
type :: DenseNN
    ! head (first_hid), tail (output) of wrapped linked list;
    ! using the simple neural network example at the top of
    ! dense_layer_definitions.f08:
    !
    !                    O ----  O  ---- O
    !
    !                   n1  w1  n2   w2  n3
    !
    ! first_hid = (w1,n2), output = (w2,n3)
    class(DenseLayer), pointer :: first_hid, output
    integer                    :: in_nodes, batch_size
    logical                    :: is_init
contains
    ! wrapper procedures that initiate DenseLayer traversals
    procedure, pass            :: dnn_add_layer, dnn_init, dnn_forw_prop, &
                                  dnn_out_delta, dnn_back_prop, dnn_update
end type
contains

!===============================================================================
! constructors / destructors
!===============================================================================

!-------------------------------------------------------------------------------
! constructs a new DenseNN
!-------------------------------------------------------------------------------
! input_nodes: (integer) input variables
!-------------------------------------------------------------------------------
! returns ::   (DenseNN pointer) new DenseNN
!-------------------------------------------------------------------------------
function create_dnn(input_nodes)
    class(DenseNN), pointer :: create_dnn
    integer, intent(in)     :: input_nodes

    allocate(create_dnn)
    create_dnn%in_nodes  =  input_nodes
    create_dnn%is_init   =  .false.
    create_dnn%first_hid => null()
    create_dnn%output    => null()
end function

!-------------------------------------------------------------------------------
! deallocate a DenseNN
!-------------------------------------------------------------------------------
! dnn:      (DenseNN pointer)
!-------------------------------------------------------------------------------
! alters :: dnn and the DenseLayers it wraps are deallocated
!-------------------------------------------------------------------------------
subroutine deallocate_dnn(dnn)
    class(DenseNN), pointer    :: dnn
    class(DenseLayer), pointer :: curr_layer

    if (dnn%is_init) then
        curr_layer => dnn%first_hid%next_layer ! second layer

        ! traverse layers, deallocating previous layer
        do while(associated(curr_layer))
            call deallocate_dense_layer(curr_layer%prev_layer)
            curr_layer => curr_layer%next_layer
        end do

        ! output layer not a 'previous' layer; manually deallocate it
        call deallocate_dense_layer(dnn%output)
        deallocate(dnn)
    end if
end subroutine

!===============================================================================
! DenseNN procedures
!   *** require input and labels in variables-as-columns form
!===============================================================================

!-------------------------------------------------------------------------------
! create and add a new DenseLayer to the tail of this DenseNN's linked list
!-------------------------------------------------------------------------------
! this:       (DenseNN - implicitly passed)
! out_nodes:  (integer) nodes output by new DenseLayer
! activation: (characters) activation function
! drop_rate:  (real) % of input nodes to dropout
!-------------------------------------------------------------------------------
! alters ::   new DenseLayer appended to this DenseNN's linked list
!-------------------------------------------------------------------------------
subroutine dnn_add_layer(this, out_nodes, activation, drop_rate)
    class(DenseNN)             :: this
    integer, intent(in)        :: out_nodes
    character(*), intent(in)   :: activation
    real(kind=8), intent(in)   :: drop_rate
    class(DenseLayer), pointer :: new_layer
    integer                    :: in_nodes

    ! make new tail matrices align with current tail matrices
    if (associated(this%output)) then
        in_nodes = this%output%out_nodes ! new tail fed by existing tail
    else
        in_nodes = this%in_nodes         ! first layer fed by input variables
    end if

    new_layer => create_dense_layer(in_nodes, out_nodes, activation, drop_rate)

    if (associated(this%output)) then
        ! new tail appended to existing tail
        this%output%next_layer => new_layer
        new_layer%prev_layer => this%output
    else
        ! adding first layer
        this%first_hid => new_layer
    end if

    this%output => new_layer
end subroutine

!-------------------------------------------------------------------------------
! initialize all DenseLayers in this DenseNN, and specify batch size to process
!-------------------------------------------------------------------------------
! this:       (DenseNN - implicitly passed)
! batch_size: (integer) examples to process (at once) before back prop
!-------------------------------------------------------------------------------
! alters ::   this DenseNN's DenseLayers are allocated and become usable
!-------------------------------------------------------------------------------
subroutine dnn_init(this, batch_size)
    class(DenseNN) :: this
    integer        :: batch_size

    this%is_init = .true.
    this%batch_size = batch_size
    call this%first_hid%dense_init(batch_size)
end subroutine

!-------------------------------------------------------------------------------
! wrapper subroutine to forward propagate input through DenseNN's DenseLayers
!-------------------------------------------------------------------------------
! this:     (DenseNN - implicitly passed)
! input:    (real(:,:)) input batch to forward propagate
! is_train: (logical) in training iteration
!-------------------------------------------------------------------------------
! alters :: this DenseNN's DenseLayers' z's and a's calculated
!-------------------------------------------------------------------------------
subroutine dnn_forw_prop(this, input, is_train)
    class(DenseNN)           :: this
    real(kind=8), intent(in) :: input(:,:)
    logical, intent(in)      :: is_train

    call this%first_hid%dense_forw_prop(input, is_train)

    ! different activation function on output layer
    ! a(l) = out_activ(z(l))
    call out_activfunc_2D(this%output%z, this%output%activ, this%output%a)
end subroutine

!-------------------------------------------------------------------------------
! calculates delta for DenseNN's output layer (derivative of loss w.r.t. z)
!-------------------------------------------------------------------------------
! this:     (DenseNN - implicitly passed)
! labels:   (real(:,:)) targets we are trying to predict
! loss:     (characters) loss function
!-------------------------------------------------------------------------------
! alters :: this DenseNN's output DenseLayer's d is calculated
!-------------------------------------------------------------------------------
subroutine dnn_out_delta(this, labels, loss)
    class(DenseNN)           :: this
    real(kind=8), intent(in) :: labels(:,:)
    character(*), intent(in) :: loss

    select case (loss)
        case ('mse')
            select case(this%output%activ)
                case ('sigmoid', 'relu', 'leaky_relu', 'elu')
                    ! d(L) = (a(L) - labels) * out_activ_deriv(z(L));
                    ! (these are implemented with element-wise derivatives)
                    this%output%d = &
                            (this%output%a - labels) * &
                            activfunc_deriv(this%output%z, this%output%activ)

                case default
                    print *, '-----------------------------------------'
                    print *, '(dense_neural_net :: dnn_out_delta)'
                    print *, 'mse - invalid output activation function.'
                    print *, 'supported: sigmoid, relu, leaky_relu, elu'
                    print *, '-----------------------------------------'
                    stop -1
                end select

        case ('cross_entropy')
            if (this%output%activ /= 'softmax') then
                print *, '---------------------------------------------------'
                print *, '(dense_neural_net :: dnn_out_delta)'
                print *, 'cross_entropy - invalid output activation function.'
                print *, 'supported: softmax'
                print *, '---------------------------------------------------'
                stop -1
            end if
            
            ! d(L) = a(L) - labels (this is fun to prove)
            this%output%d = this%output%a - labels

        case default
            print *, '-----------------------------------'
            print *, '(dense_neural_net :: dnn_out_delta)'
            print *, 'invalid loss function.'
            print *, 'supported: mse, cross_entropy'
            print *, '-----------------------------------'
            stop -1
    end select
end subroutine

!-------------------------------------------------------------------------------
! wrapper subroutine to back propagate through DenseNN's DenseLayers
!-------------------------------------------------------------------------------
! this:     (DenseNN - implicitly passed)
! labels:   (real(:,:)) targets we are trying to predict
! loss:     (characters) loss function
!-------------------------------------------------------------------------------
! alters :: this DenseNN's DenseLayers' d's calculated
!-------------------------------------------------------------------------------
subroutine dnn_back_prop(this, labels, loss)
    class(DenseNN)           :: this
    real(kind=8), intent(in) :: labels(:,:)
    character(*), intent(in) :: loss

    call this%dnn_out_delta(labels, loss)

    if (associated(this%output%prev_layer)) then
        call this%output%prev_layer%dense_back_prop()
    end if
end subroutine

!-------------------------------------------------------------------------------
! wrapper subroutine to adjust weights and biases in DenseNN's DenseLayers
!-------------------------------------------------------------------------------
! this:       (DenseNN - implicitly passed)
! input:      (real(:,:)) input batch
! learn_rate: (real) scale factor for change in weights and biases
! is_train:   (logical) in training iteration
!-------------------------------------------------------------------------------
! alters ::   this DenseNN's weights and biases adjusted to minimize loss
!-------------------------------------------------------------------------------
subroutine dnn_update(this, input, learn_rate, is_train)
    class(DenseNN)           :: this
    real(kind=8), intent(in) :: input(:,:), learn_rate
    logical, intent(in)      :: is_train

    ! first_hid's a(l-1) is input batch
    call this%first_hid%dense_update(input, learn_rate, is_train)
end subroutine
end module
