!-------------------------------------------------------------------------------
! implementation for pooling layer type (PoolLayer)
!-------------------------------------------------------------------------------

module pool_layer_definitions
use net_helper_procedures
implicit none

!===============================================================================
! types
!===============================================================================

! represents a Pooling Layer in a neural network
!
! this keeps track of the shape of the input to the PoolLayer, the pooled
! outputs, and also the indices in the input that correspond to the values
! used in the pooled output:
!
! the pooled outputis a 4D array with dimensions:
! (height, width, channel, batch_item)
!
! we only compare about the (height, width) of the kernel that passes over the
! input, because there are no kernel values being used (just used for selecting
! a window)
!
! the array corresponding to indices in a has deimsnsions:
! (height, width, (2: Y, X), channel, batch_item)
! at each height/width, there is then an array of size 2, which stores the
! (Y, X) coordinate of the value in the input array which corresponds to the
! value used at (height, width) in the pooled output;
! both (Y, X) will be -1 if the used value was within padding, and not the array
!
! currently, there is only max pooling implementation, but mean pooling can be
! implemented in pool_forw_prop, pool_back_prop
type :: PoolLayer
    integer                   :: in_dims(3), k_dims(2), stride(2), &
                                 out_rows, out_cols, out_channels, out_count, &
                                 batch_size
    character(len=20)         :: pool, pad
    ! dimensions: (height, width, channel, batch_item)
    real(kind=8), allocatable :: a(:,:,:,:)        ! pool result
    ! dimensions: (height, width, (2: Y, X), channel, batch_item)
    integer, allocatable      :: a_idxs(:,:,:,:,:) ! indices for pool result
contains
    ! procedures that implement basic pooling functionality 
    procedure, pass           :: pool_init, pool_forw_prop, pool_back_prop
end type
contains

!===============================================================================
! constructors / destructors
!===============================================================================

!-------------------------------------------------------------------------------
! constructs a new PoolLayer
!-------------------------------------------------------------------------------
! input_dims:  (integer(3)) (height, width, channels) of one input
! kernel_dims: (integer(2)) (height, width) of pool kernel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! pool:        (characters) pool type
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! returns ::   (PoolLayer pointer) new PoolLayer
!-------------------------------------------------------------------------------
function create_pool_layer(input_dims, kernel_dims, stride, pool, padding)
    class(PoolLayer), pointer :: create_pool_layer
    integer, intent(in)       :: input_dims(3), kernel_dims(2), stride(2)
    character(*), intent(in)  :: pool, padding
    integer                   :: pad_rows, final_rows, pad_cols, final_cols

    allocate(create_pool_layer)
    create_pool_layer%in_dims = input_dims
    create_pool_layer%k_dims  = kernel_dims
    create_pool_layer%stride  = stride
    create_pool_layer%pool    = pool
    create_pool_layer%pad     = padding

    ! determine output shape from input pad then applying kernel
    pad_rows   = pad_calc(input_dims(1), kernel_dims(1), stride(1), padding)
    pad_cols   = pad_calc(input_dims(2), kernel_dims(2), stride(2), padding)
    final_rows = res_calc(input_dims(1)+pad_rows, kernel_dims(1), stride(1))
    final_cols = res_calc(input_dims(2)+pad_cols, kernel_dims(2), stride(2))

    create_pool_layer%out_rows     = final_rows
    create_pool_layer%out_cols     = final_cols
    create_pool_layer%out_channels = input_dims(3) ! channels not altered

    ! outward facing node count
    create_pool_layer%out_count = create_pool_layer%out_rows * &
                                  create_pool_layer%out_cols * &
                                  create_pool_layer%out_channels
end function

!-------------------------------------------------------------------------------
! deallocate a PoolLayer
!-------------------------------------------------------------------------------
! l:        (PoolLayer pointer)
!-------------------------------------------------------------------------------
! alters :: l is deallocated
!-------------------------------------------------------------------------------
subroutine deallocate_pool_layer(l)
    class(PoolLayer), pointer :: l
    deallocate(l%a, l%a_idxs)
    deallocate(l)
end subroutine

!===============================================================================
! PoolLayer procedures
!===============================================================================

!-------------------------------------------------------------------------------
! initialize the various matrices of PoolLayer
!-------------------------------------------------------------------------------
! this:       (PoolLayer - implicitly passed)
! batch_size: (integer) examples to process before back prop
!-------------------------------------------------------------------------------
! alters ::   this PoolLayer's matrices are allocated and become usable
!-------------------------------------------------------------------------------
subroutine pool_init(this, batch_size)
    class(PoolLayer)    :: this
    integer, intent(in) :: batch_size

    this%batch_size = batch_size

    allocate(this%a(this%out_rows, this%out_cols, this%out_channels, &
                    batch_size), &
             this%a_idxs(this%out_rows, this%out_cols, 2, this%out_channels, &
                    batch_size))
end subroutine

!-------------------------------------------------------------------------------
! forward propagate through PoolLayer
!-------------------------------------------------------------------------------
! this:     (PoolLayer - implicitly passed)
! input:    (real(:,:,:,:)) input batch to forward propagate
!-------------------------------------------------------------------------------
! alters :: this PoolLayer's a and a_idxs are calculated
!-------------------------------------------------------------------------------
subroutine pool_forw_prop(this, input)
    class(PoolLayer)          :: this
    real(kind=8), intent(in)  :: input(:,:,:,:)
    real(kind=8), allocatable :: a_slice(:,:,:)
    integer, allocatable      :: a_idxs_slice(:,:,:,:)
    integer                   :: i

    if (this%pool == 'max') then
        ! max pool each batch item
        do i = 1, this%batch_size
            call max_pool_3D(input(:,:,:,i), this%pad, this%k_dims, &
                             this%stride, a_slice, a_idxs_slice)

            this%a(:,:,:,i) = a_slice
            this%a_idxs(:,:,:,:,i) = a_idxs_slice
        end do
    else
        print *, '------------------------------------------'
        print *, '(pool_layer_definitions :: pool_forw_prop)'
        print *, 'invalid pooling type.'
        print *, 'supported: max'
        print *, '------------------------------------------'
        stop -1
    end if
end subroutine

!-------------------------------------------------------------------------------
! 'backpropagate' deltas corresponding to the pooled output array, wrt the
! corresponding values in the array before the pool for one batch item
!
! this currently only functions on one 3D batch item within 4D res, in order to
! reflect the helper functions in net_helper_procedures; pool_back_prop will
! only be called on individual batch_items for now due to simplicity, but it
! is possible to adjust the callers to this procedure to process an entire batch
! (would require an additional pool_back_prop procedure to handle the change)
!-------------------------------------------------------------------------------
! delta:      (real(:,:,:)) deltas; shape corresponding to pooled output array
! batch_item: (integer) index of item in batch
! res:        (real(:,:,:,:)) stores values; shape corresponding to input array
!-------------------------------------------------------------------------------
! alters ::    res stores the deltas corresponding to the array before the pool
!-------------------------------------------------------------------------------
subroutine pool_back_prop(this, delta, batch_item, res)
    class(PoolLayer)          :: this
    real(kind=8), intent(in)  :: delta(:,:,:)
    integer, intent(in)       :: batch_item
    real(kind=8), allocatable :: res(:,:,:,:)
    integer                   :: row, col, chan, max_r, max_c

    if (.not. all(shape(delta) == [this%out_rows, this%out_cols, &
                                   this%out_channels])) then
        print *, '------------------------------------------'
        print *, '(pool_layer_definitions :: pool_back_prop)'
        print *, 'invalid delta shape.'
        print *, '------------------------------------------'
        stop -1
    end if

    if (allocated(res)) then
        if (.not. all(shape(res) == [this%in_dims(1), this%in_dims(2), &
                                     this%in_dims(3), this%batch_size])) then
            ! some logic behind calling this is messed up; fail outright
            print *, '------------------------------------------'
            print *, '(pool_layer_definitions :: pool_back_prop)'
            print *, 'invalid res shape.'
            print *, '------------------------------------------'
            stop -1
        end if
    else
        allocate(res(this%in_dims(1), this%in_dims(2), this%in_dims(3), &
                     this%batch_size))
    end if

    res(:,:,:,batch_item) = 0

    ! for each delta (max), increment index of res that corresponds to the max
    do row = 1, this%out_rows
        do col = 1, this%out_cols
            do chan = 1, this%out_channels
                max_r = this%a_idxs(row,col,1,chan,batch_item)

                ! if the max index was not in the padding
                if (max_r /= -1) then
                    max_c = this%a_idxs(row,col,2,chan,batch_item)
                    
                    ! add delta wrt corresponding max a
                    res(max_r,max_c,chan,batch_item) = &
                                            res(max_r,max_c,chan,batch_item) + &
                                            delta(row,col,chan)
                end if
            end do
        end do
    end do
end subroutine
end module
