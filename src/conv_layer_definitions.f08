!-------------------------------------------------------------------------------
! implementation for ConvLayer (convolutional) type (see below)
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module conv_layer_definitions
use net_helper_procedures
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

! represents a Convolutional Layer in a neural network
!
! the ConvLayer keeps track of the kernels that pass over (cross-correlate with)
! the input channels, and the biases, weighted inputs, activations, and deltas
! (errors) that are used on the result of passing the kernels over the input
! channels
!
! each kernel is shared between images in a batch, and each produces an output
! channel. kernels have 4D matrix dimensions: (rows, columns, channels, batches)
!
! the implementation here uses 'untied' biases, whereby the bias comprises
! multiple nodes to match the output channel produced by passing a kernel over
! an input. there is one copy that is shared between images. biases have 3D
! matrix dimensions: (rows, columns, channels)
!
! weighted inputs, activations, and deltas match the output channel produced by
! passing a kernel over an input, and are passed between ConvLayers to preserve
! batch information. each have 4D matrices with dimensions:
! (rows, columns, channels, batches)
!
! an input image can have any number of channels (1 for grayscale, 3 for RGB);
! the third dimension for all of the matrices is the channels, so no special
! processing must be done; an input must just be specified with 3 channels
!
! ConvLayers are connected to each other as a doubly-linked list, and each
! ConvLayer can have a PoolLayer that it feeds into before reaching the next
! ConvLayer
type :: ConvLayer
    integer                   :: in_dims(3), k_dims(2), stride(2), &
                                 out_rows, out_cols, out_channels, out_count, &
                                 batch_size
    class(ConvLayer), pointer :: prev_layer, next_layer
    class(PoolLayer), pointer :: next_pool
    character(len=20)         :: pad, activ
    ! dimension order: (rows, columns, channels, batches or kernel items)
    real, allocatable         :: k(:,:,:,:), & ! kernels
                                 b(:,:,:),   & ! biases
                                 z(:,:,:,:), & ! weighted inputs
                                 a(:,:,:,:), & ! activations
                                 d(:,:,:,:)    ! deltas (errors)
contains
    ! procedures that traverse through linked list of ConvLayers
    procedure, pass           :: conv_init, conv_add_pool_layer, &
                                 conv_forw_prop, conv_back_prop, conv_update
end type
contains 

!===============================================================================
! constructors / destructors
!===============================================================================

!-------------------------------------------------------------------------------
! constructs a new ConvLayer;
!
! 'same' padding maintains dimensions;
! 'valid' padding reduces dimensions;
! 'full' padding applies opposite operations to 'valid'; upscaling 
!-------------------------------------------------------------------------------
! input_dims:  (integer(3)) (height, width, channels) of one input
! kernels:     (integer) kernels in this layer
! kernel_dims: (integer(2)) (height, width) of kernel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! activ:       (characters) activation function
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! returns ::   (ConvLayer pointer) new ConvLayer
!-------------------------------------------------------------------------------
function create_conv_layer(input_dims, kernels, kernel_dims, stride, &
                           activ, padding)
    class(ConvLayer), pointer :: create_conv_layer
    integer, intent(in)       :: input_dims(3), kernels, kernel_dims(2), &
                                 stride(2)
    character(*), intent(in)  :: activ, padding
    integer                   :: pad_rows, final_rows, pad_cols, final_cols

    if (.not. (padding == 'valid' .or. &
               padding == 'same' .or. &
               padding == 'full')) then
        print *, '---------------------------------------------'
        print *, '(conv_layer_definitions :: create_conv_layer)'
        print *, 'supported: valid, same, full'
        print *, '---------------------------------------------'
        stop -1
    end if

    allocate(create_conv_layer)
    create_conv_layer%in_dims    =  input_dims
    create_conv_layer%k_dims     =  kernel_dims
    create_conv_layer%stride     =  stride
    create_conv_layer%activ      =  activ
    create_conv_layer%pad        =  padding
    create_conv_layer%prev_layer => null()
    create_conv_layer%next_layer => null()
    create_conv_layer%next_pool  => null()

    ! determine output shape from input pad then applying kernel
    pad_rows   = pad_calc(input_dims(1), kernel_dims(1), stride(1), padding)
    pad_cols   = pad_calc(input_dims(2), kernel_dims(2), stride(2), padding)
    final_rows = res_calc(input_dims(1)+pad_rows, kernel_dims(1), stride(1))
    final_cols = res_calc(input_dims(2)+pad_cols, kernel_dims(2), stride(2))

    create_conv_layer%out_rows     = final_rows
    create_conv_layer%out_cols     = final_cols
    create_conv_layer%out_channels = kernels

    ! outward facing node count
    create_conv_layer%out_count = create_conv_layer%out_rows * &
                                  create_conv_layer%out_cols * &
                                  create_conv_layer%out_channels
end function

!-------------------------------------------------------------------------------
! deallocate a ConvLayer
!-------------------------------------------------------------------------------
! l:        (ConvLayer pointer)
!-------------------------------------------------------------------------------
! alters :: l is deallocated
!-------------------------------------------------------------------------------
subroutine deallocate_conv_layer(l)
    class(ConvLayer), pointer :: l

    if (associated(l%next_pool)) then
        call deallocate_pool_layer(l%next_pool) ! deallocate trailing pool layer
    end if

    deallocate(l%k, l%b, l%z, l%a, l%d)
    deallocate(l)
end subroutine

!===============================================================================
! ConvLayer procedures
!===============================================================================

!-------------------------------------------------------------------------------
! add a PoolLayer for this ConvLayer to feed into
!-------------------------------------------------------------------------------
! this:        (ConvLayer - implicitly passed)
! kernel_dims: (integer(2)) (height, width) of the pool kernel
! stride:      (integer(2)) size of pool kernel moves in (y, x) directions
! pool:        (characters) pool type
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! alters ::    this ConvLayer now has a next_pool PoolLayer
!-------------------------------------------------------------------------------
subroutine conv_add_pool_layer(this, kernel_dims, stride, pool, padding)
    class(ConvLayer)         :: this
    integer, intent(in)      :: kernel_dims(2), stride(2)
    character(*), intent(in) :: pool, padding

    ! feed this ConvLayers activations (output) into PoolLayer
    this%next_pool => create_pool_layer([this%out_rows, this%out_cols, &
                                         this%out_channels], &
                                        kernel_dims, stride, pool, padding)
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to initialize the various matrices of ConvLayers
!-------------------------------------------------------------------------------
! this:       (ConvLayer - implicitly passed)
! batch_size: (integer) examples to process before back prop
!-------------------------------------------------------------------------------
! alters ::   this ConvLayer's matrices are allocated and become usable
!-------------------------------------------------------------------------------
subroutine conv_init(this, batch_size)
    class(ConvLayer)    :: this
    integer, intent(in) :: batch_size

    this%batch_size = batch_size

    allocate(this%k(this%k_dims(1), this%k_dims(2), this%in_dims(3), &
                    this%out_channels), &
             this%b(this%out_rows, this%out_cols, this%out_channels), &

             this%z(this%out_rows, this%out_cols, this%out_channels, &
                    batch_size), &
             this%a(this%out_rows, this%out_cols, this%out_channels, &
                    batch_size), &
             this%d(this%out_rows, this%out_cols, this%out_channels, &
                    batch_size))

    ! initialize kernels and biases
    call random_number(this%k)   ! random weights, uniform range: [0,1)
    this%k = (this%k - 0.5) / 10 ! shift and scale to [-0.05, 0.05]
    this%b = 0

    ! init next PoolLayer
    if (associated(this%next_pool)) then
        call this%next_pool%pool_init(batch_size)
    end if

    ! traverse next ConvLayers
    if (associated(this%next_layer)) then
        call this%next_layer%conv_init(batch_size)
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to forward propagate through ConvLayers
!-------------------------------------------------------------------------------
! this:     (ConvLayer - implicitly passed)
! input:    (real(:,:,:,:)) input batch to forward propagate
!-------------------------------------------------------------------------------
! alters :: this ConvLayer's z and a are calculated
!-------------------------------------------------------------------------------
subroutine conv_forw_prop(this, input)
    class(ConvLayer)  :: this
    real, intent(in)  :: input(:,:,:,:)
    real, allocatable :: z_slice(:,:,:)
    integer           :: i

    do i = 1, this%batch_size
        if (this%pad == 'full') then
            ! z(l) = transpose_convolution(a(l-1), k(l)) + b(l)
            call transpose_convolve_3D_kernels(input(:,:,:,i), this%pad, &
                                               this%k, this%stride, z_slice)
        else
            ! z(l) = cross_correlation(a(l-1), k(l)) + b(l)
            call cross_correlate_3D_kernels(input(:,:,:,i), this%pad, this%k, &
                                            this%stride, z_slice)
        end if

        this%z(:,:,:,i) = z_slice + this%b
    end do

    ! output activation cannot be softmax (or any "special case" function),
    ! so we can call all activations in the same way
    ! a(l) = activ(z(l))
    this%a = activfunc(this%z, this%activ)

    ! forward prop activations through pool
    if (associated(this%next_pool)) then
        call this%next_pool%pool_forw_prop(this%a)
    end if

    ! traverse next layers
    if (associated(this%next_layer)) then
        if (associated(this%next_pool)) then
            ! has next_pool; forward prop pooled output
            call this%next_layer%conv_forw_prop(this%next_pool%a)
        else
            ! forward prop regular output
            call this%next_layer%conv_forw_prop(this%a)
        end if
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to back propagate through ConvLayers
!-------------------------------------------------------------------------------
! this:     (ConvLayer - implicitly passed)
!-------------------------------------------------------------------------------
! alters :: this ConvLayer's d is calculated
!-------------------------------------------------------------------------------
subroutine conv_back_prop(this)
    class(ConvLayer)  :: this
    real, allocatable :: d_slice(:,:,:)
    integer           :: i

    if (.not. associated(this%next_layer)) then
        print *, '-------------------------------------------'
        print *, '(conv_layer_definitions :: conv_back_prop)'
        print *, 'cannot call conv_back_prop on output layer.'
        print *, '-------------------------------------------'
        stop -1
    end if

    do i = 1, this%batch_size
        if (this%next_layer%pad == 'full') then
            call cross_correlate_3D_perms_sum_kernel( &
                                                this%next_layer%d(:,:,:,i), &
                                                this%next_layer%pad, &
                                                this%next_layer%k, &
                                                this%next_layer%stride, d_slice)
        else
            ! derivative d(l+1) wrt a(l) OR pool(l) (if present) 
            call transpose_convolve_3D_perms_sum_kernel( &
                                                this%next_layer%d(:,:,:,i), &
                                                this%next_layer%pad, &
                                                this%next_layer%k, &
                                                this%next_layer%stride, d_slice)
        end if

        if (associated(this%next_pool)) then
            ! must undo the pooling to find derivative wrt a's kept by pool
            call this%next_pool%pool_back_prop(d_slice, i, this%d)
        else
            ! no pooling to process
            this%d(:,:,:,i) = d_slice
        end if
    end do

    ! derivative wrt z
    this%d = this%d * activfunc_deriv(this%z, this%activ)

    ! traverse previous layers
    if (associated(this%prev_layer)) then
        call this%prev_layer%conv_back_prop()
    end if
end subroutine

!-------------------------------------------------------------------------------
! helper subroutine to adjust kernels and biases in ConvLayers
!-------------------------------------------------------------------------------
! this:       (ConvLayer - implicitly passed)
! input:      (real(:,:,:,:)) previous layer activations
! learn_rate: (real) scale factor for change in kernels and biases
!-------------------------------------------------------------------------------
! alters ::   this ConvLayer's kernels and biases adjusted to minimize loss
!-------------------------------------------------------------------------------
subroutine conv_update(this, input, learn_rate)
    class(ConvLayer)  :: this
    real, intent(in)  :: input(:,:,:,:), learn_rate
    real, allocatable :: total_k_change(:,:,:,:), k_change(:,:,:,:)
    real              :: scale
    integer           :: i

    ! initial call to start counter array (total_k_change)
    if (this%pad == 'full') then
        call cross_correlate_3D_perms_group_base( &
                                            this%d(:,:,:,1), this%pad, &
                                            input(:,:,:,1), this%stride, &
                                            'left', total_k_change)
    else
        call cross_correlate_3D_perms_group_kernel( &
                                            input(:,:,:,1), this%pad, &
                                            this%d(:,:,:,1), this%stride, &
                                            'right', total_k_change)
    end if
    
    ! sum kernel changes across batch
    do i = 2, this%batch_size
        if (this%pad == 'full') then
            call cross_correlate_3D_perms_group_base( &
                                                this%d(:,:,:,i), this%pad, &
                                                input(:,:,:,i), this%stride, &
                                                'left', k_change)
        else
            call cross_correlate_3D_perms_group_kernel( &
                                                input(:,:,:,i), this%pad, &
                                                this%d(:,:,:,i), this%stride, &
                                                'right', k_change)
        end if

        total_k_change = total_k_change + k_change
    end do

    ! multiply by learn_rate, then average across all examples in batch
    scale = learn_rate / this%batch_size

    ! weights updated on average change
    this%k = this%k - total_k_change * scale

    ! biases updated on average batch deltas
    this%b = this%b - sum(this%d, dim=4) * scale

    ! traverse next layers
    if (associated(this%next_layer)) then
        if (associated(this%next_pool)) then
            ! prop pooled output
            call this%next_layer%conv_update(this%next_pool%a, learn_rate)
        else
            call this%next_layer%conv_update(this%a, learn_rate)
        end if
    end if
end subroutine
end module