!-------------------------------------------------------------------------------
! implementation of a convolutional and deconvolutional layer type (ConvLayer)
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
! the implementation here uses 'tied' biases, where there is 1 bias per channel;
! bias value repeated into higher dimension array so we can add array later
! rather than using nested loops
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
    real(kind=8), allocatable :: k(:,:,:,:), & ! kernels
                                 b(:,:,:),   & ! biases
                                 z(:,:,:,:), & ! weighted inputs
                                 a(:,:,:,:), & ! activations
                                 d(:,:,:,:), & ! deltas (errors)
                                 drop(:,:,:,:) ! dropout inputs (0=drop, 1=keep)
    real                      :: drop_rate     ! % of input nodes to drop
contains
    ! procedures that traverse through linked list of ConvLayers
    procedure, pass           :: conv_init, conv_add_pool_layer, &
                                 conv_forw_prop, conv_back_prop, conv_update, &
                                 conv_dropout_rand
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
! drop_rate:   (real) % of input nodes to dropout
!-------------------------------------------------------------------------------
! returns ::   (ConvLayer pointer) new ConvLayer
!-------------------------------------------------------------------------------
function create_conv_layer(input_dims, kernels, kernel_dims, stride, &
                           activ, padding, drop_rate)
    class(ConvLayer), pointer :: create_conv_layer
    integer, intent(in)       :: input_dims(3), kernels, kernel_dims(2), &
                                  stride(2)
    character(*), intent(in)  :: activ, padding
    real, intent(in)          :: drop_rate
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
    create_conv_layer%drop_rate  =  drop_rate
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
                    batch_size), &

             this%drop(this%in_dims(1), this%in_dims(2), this%in_dims(3), &
                       batch_size))

    ! initialize kernels and biases
    call random_number(this%k)   ! random weights, uniform range: [0,1)
    this%k = (this%k - 0.5) / 10 ! shift and scale to [-0.05, 0.05)
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
! helper subroutine to random initialize/overwrite a ConvLayers dropout array
!-------------------------------------------------------------------------------
! this:       (ConvLayer - implicitly passed)
!-------------------------------------------------------------------------------
! alters ::   this ConvLayer's drop array has randomized values
!-------------------------------------------------------------------------------
subroutine conv_dropout_rand(this)
    class(ConvLayer)   :: this

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
! helper subroutine to forward propagate through ConvLayers
!-------------------------------------------------------------------------------
! this:     (ConvLayer - implicitly passed)
! input0:   (real(:,:,:,:)) input batch to forward propagate
! is_train: (logical) in training iteration
!-------------------------------------------------------------------------------
! alters :: this ConvLayer's z and a are calculated
!-------------------------------------------------------------------------------
subroutine conv_forw_prop(this, input0, is_train)
    class(ConvLayer)          :: this
    real(kind=8), intent(in)  :: input0(:,:,:,:)
    logical, intent(in)       :: is_train
    real(kind=8), allocatable :: input(:,:,:,:), z_slice(:,:,:)
    integer                   :: i

    if (this%drop_rate > 0 .and. is_train) then
        call this%conv_dropout_rand() ! randomize dropout
        input = input0 * this%drop
    else
        input = input0
    end if

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
            call this%next_layer%conv_forw_prop(this%next_pool%a, is_train)
        else
            ! forward prop regular output
            call this%next_layer%conv_forw_prop(this%a, is_train)
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
    class(ConvLayer)          :: this
    real(kind=8), allocatable :: d_slice(:,:,:)
    integer                   :: i

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
! input0:     (real(:,:,:,:)) previous layer activations
! learn_rate: (real) scale factor for change in kernels and biases
! is_train:   (logical) in training iteration
!-------------------------------------------------------------------------------
! alters ::   this ConvLayer's kernels and biases adjusted to minimize loss
!-------------------------------------------------------------------------------
subroutine conv_update(this, input0, learn_rate, is_train)
    class(ConvLayer)          :: this
    real(kind=8), intent(in)  :: input0(:,:,:,:)
    real, intent(in)          :: learn_rate
    logical, intent(in)       :: is_train
    real(kind=8), allocatable :: input(:,:,:,:), total_k_change(:,:,:,:), &
                                 k_change(:,:,:,:), chan_avgs(:,:,:), &
                                 chan_biases(:)
    real                      :: scale
    integer                   :: i

    if (this%drop_rate > 0 .and. is_train) then
        call this%conv_dropout_rand() ! randomize dropout
        input = input0 * this%drop
    else
        input = input0
    end if

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

    ! calculate 1 bias per channel per batch;
    ! start with 4D: (row,col,channel,batch):

    ! sum then average by batch, and scale, 3D: (row,col,channel)
    chan_avgs = sum(this%d, dim=4) * scale

    ! sum then average channels, 1D: (channel), where each val is a bias
    chan_biases = sum(sum(chan_avgs, dim=1), dim=1) / &       ! sum
                  (size(this%d, dim=1) * size(this%d, dim=2)) ! count

    ! one bias per channel -> 2D array; can add bias array instead of loops
    do i = 1, size(this%b, dim=3)
        this%b(:,:,i) = this%b(:,:,i) - chan_biases(i)
    end do

    ! traverse next layers
    if (associated(this%next_layer)) then
        if (associated(this%next_pool)) then
            ! prop pooled output
            call this%next_layer%conv_update(this%next_pool%a, &
                                             learn_rate, is_train)
        else
            call this%next_layer%conv_update(this%a, learn_rate, is_train)
        end if
    end if
end subroutine
end module
