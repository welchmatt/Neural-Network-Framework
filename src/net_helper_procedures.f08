!-------------------------------------------------------------------------------
! helper functions for matrices; utilized by dense_neural_net, conv_neural_net,
! and sequential_neural_net, but can be used for matrices in general
!-------------------------------------------------------------------------------
! Matt Welch
!-------------------------------------------------------------------------------

module net_helper_procedures
implicit none
contains

!===============================================================================
!===============================================================================
! procedures with 2D array input require variables-as-columns form
!===============================================================================
!===============================================================================

!===============================================================================
! activation functions and their derivatives
!   *** elemental functions can also be applied to arrays of the input type
!===============================================================================

!-------------------------------------------------------------------------------
! sigmoid activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function sigmoid(z)
    real, intent(in) :: z
    sigmoid = exp(z) / (exp(z) + 1)
end function

!-------------------------------------------------------------------------------
! derivative of sigmoid activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function sigmoid_deriv(z)
    real, intent(in) :: z
    sigmoid_deriv = exp(z) / (exp(z) + 1)**2
end function

!-------------------------------------------------------------------------------
! relu activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function relu(z)
    real, intent(in) :: z
    if (z >= 0) then
        relu = z
    else
        relu = 0
    end if
end function

!-------------------------------------------------------------------------------
! derivative of relu activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function relu_deriv(z)
    real, intent(in) :: z
    if (z > 0) then
        relu_deriv = 1
    else
        relu_deriv = 0
    end if
end function

!-------------------------------------------------------------------------------
! leaky relu activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function leaky_relu(z)
    real, intent(in) :: z
    if (z > 0) then
        leaky_relu = z
    else
        leaky_relu = 0.01 * z
    end if
end function

!-------------------------------------------------------------------------------
! derivative of leaky relu activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function leaky_relu_deriv(z)
    real, intent(in) :: z
    if (z > 0) then
        leaky_relu_deriv = 1
    else
        leaky_relu_deriv = 0.01
    end if
end function

!-------------------------------------------------------------------------------
! elu activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function elu(z)
    real, intent(in) :: z
    if (z > 0) then
        elu = z
    else
        elu = 0.01 * (exp(z) - 1)
    end if
end function

!-------------------------------------------------------------------------------
! derivative of elu activation function
!-------------------------------------------------------------------------------
! z:         (real)
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
elemental real function elu_deriv(z)
    real, intent(in) :: z
    if (z > 0) then
        elu_deriv = 1
    else
        elu_deriv = 0.01 * exp(z)
    end if
end function

!-------------------------------------------------------------------------------
! softmax activation function (applied to rows)
!-------------------------------------------------------------------------------
! z:        (real(:,:))
!-------------------------------------------------------------------------------
! alters :: softmax occurs in-place on input a
!-------------------------------------------------------------------------------
subroutine softmax(z)
    real    :: z(:,:)
    integer :: r
    z = exp(z)
    do r = 1, size(z, dim=1)
        ! vals in each row / sum of that row
        z(r,:) = z(r,:) / sum(z(r,:))
    end do
end subroutine

!-------------------------------------------------------------------------------
! wrapper for element-wise activation functions
!-------------------------------------------------------------------------------
! z:         (real)
! activ:     (characters) activation function
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
impure elemental real function activfunc(z, activ)
    real, intent(in)         :: z
    character(*), intent(in) :: activ

    select case(activ)
        case ('sigmoid')
            activfunc = sigmoid(z)
        case ('relu')
            activfunc = relu(z)
        case ('leaky_relu')
            activfunc = leaky_relu(z)
        case ('elu')
            activfunc = elu(z)
        case default
            print *, '-----------------------------------------'
            print *, '(net_helper_functions :: activfunc)'
            print *, 'invalid activation function.'
            print *, 'supported: sigmoid, relu, leaky_relu, elu'
            print *, '-----------------------------------------'
            stop -1
    end select
end function

!-------------------------------------------------------------------------------
! wrapper for element-wise activation function derivative
!-------------------------------------------------------------------------------
! z:         (real)
! activ:     (characters) activation function
!-------------------------------------------------------------------------------
! returns :: (real)
!-------------------------------------------------------------------------------
impure elemental real function activfunc_deriv(z, activ)
    real, intent(in)         :: z
    character(*), intent(in) :: activ

    select case(activ)
        case ('sigmoid')
            activfunc_deriv = sigmoid_deriv(z)
        case ('relu')
            activfunc_deriv = relu_deriv(z)
        case ('leaky_relu')
            activfunc_deriv = leaky_relu_deriv(z)
        case ('elu')
            activfunc_deriv = elu_deriv(z)
        case default
            print *, '-----------------------------------------'
            print *, '(net_helper_functions :: activfunc_deriv)'
            print *, 'invalid activation function.'
            print *, 'supported: sigmoid, relu, leaky_relu, elu'
            print *, '-----------------------------------------'
            stop -1
    end select
end function

!-------------------------------------------------------------------------------
! wrapper for output-layer activation functions;
! subroutine needed because some output activation functions (like softmax) are
! not element-wise, so we must manipulate an array, not return a value
!-------------------------------------------------------------------------------
! a:         (real(:,:))
! out_activ: (characters) output activation function
! res:       (real(:,:)) stores the output
!-------------------------------------------------------------------------------
! alters ::  res becomes activation applied to z
!-------------------------------------------------------------------------------
subroutine out_activfunc(z, out_activ, res)
    real                     :: z(:,:)
    character(*), intent(in) :: out_activ
    real, allocatable        :: res(:,:)
    integer                  :: z_rows, z_cols

    z_rows = size(z, dim=1)
    z_cols = size(z, dim=2)

    ! create new array if not correct size
    if (allocated(res)) then
        if (.not. all(shape(res) == shape(z))) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(z_rows, z_cols))
    end if

    select case(out_activ)
        case ('softmax')
            res = z
            call softmax(res)
        case ('sigmoid')
            res = sigmoid_deriv(z)
        case ('relu')
            res = relu_deriv(z)
        case ('leaky_relu')
            res = leaky_relu_deriv(z)
        case ('elu')
            res = elu_deriv(z)
        case default
            print *, '--------------------------------------------------'
            print *, '(net_helper_functions :: out_activfunc)'
            print *, 'invalid output activation function.'
            print *, 'supported: softmax, sigmoid, relu, leaky_relu, elu'
            print *, '--------------------------------------------------'
            stop -1
    end select
end subroutine

!===============================================================================
! loss/metric functions
!===============================================================================

!-------------------------------------------------------------------------------
! mean square error between predictions and true labels
!-------------------------------------------------------------------------------
! preds:      (real(:,:)) predictions
! labels:     (real(:,:)) targets we want to predict
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function mse_func(preds, labels)
    real, intent(in) :: preds(:,:), labels(:,:)
    mse_func = sum((preds - labels) ** 2) / (2 * size(preds, dim=1))
end function

!-------------------------------------------------------------------------------
! categorical cross entropy between predictions and true labels
!-------------------------------------------------------------------------------
! preds:      (real(:,:)) predictions
! labels:     (real(:,:)) targets we want to predict
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function cross_entropy_func(preds, labels)
    real, intent(in) :: preds(:,:), labels(:,:)
    cross_entropy_func = -sum(labels * log(preds)) / size(preds, dim=1)
end function

!-------------------------------------------------------------------------------
! calculate the accuracy between predictions and one-hot label rows
!-------------------------------------------------------------------------------
! preds:      (real(:,:)) predictions
! labels:     (real(:,:)) ONE-HOT ENCODED targets we want to predict
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function one_hot_accuracy(preds, labels)
    real, intent(in) :: preds(:,:), labels(:,:)
    real             :: correct

    ! correct where strongest predictions match one-hot labels
    correct = count(maxloc(preds, dim=2) == maxloc(labels, dim=2))
    one_hot_accuracy = correct / size(preds, dim=1)
end function

!-------------------------------------------------------------------------------
! wrapper for loss functions
!-------------------------------------------------------------------------------
! preds:      (real(:,:)) predictions
! labels:     (real(:,:)) targets we want to predict
! loss:       (characters) loss function
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function lossfunc(preds, labels, loss)
    real, intent(in)         :: preds(:,:), labels(:,:)
    character(*), intent(in) :: loss

    select case(loss)
        case ('mse')
            lossfunc = mse_func(preds, labels)
        case ('cross_entropy')
            lossfunc = cross_entropy_func(preds, labels)
        case default
            print *, '----------------------------------'
            print *, '(net_helper_functions :: lossfunc)'
            print *, 'invalid loss function.'
            print *, 'supported: mse, cross_entropy'
            print *, '----------------------------------'
            stop -1
    end select
end function

!===============================================================================
! more complex array-based helpers
!===============================================================================

!-------------------------------------------------------------------------------
! shuffle rows of two 2D arrays in corresponding order (Fisher-Yates shuffle)
!-------------------------------------------------------------------------------
! a:         (real(:,:))
! b:         (real(:,:))
!-------------------------------------------------------------------------------
! alters ::  rows of a and b are shuffled (correspondingly) in-place
!-------------------------------------------------------------------------------
subroutine pair_shuffle_rows_2D(a, b)
    real              :: a(:,:), b(:,:), randn
    real, allocatable :: row(:)
    integer           :: i, j

    ! loop through rows from high to low indices
    do i = size(a, dim=1), 2, -1
        j = i + 1 ! arbitrary assignment to enter loop below

        ! generate random j in range [1,i]
        do while (j > i)
            call random_number(randn) ! range [0, 1)
            ! scale to [0, i) => floor to [0,i-1] => shift to [1,i]
            j = 1 + floor(randn * i)
        end do

        ! swap if random index in different place
        ! (if random j = i, don't need to swap i with i)
        if (i /= j) then
            ! array a
            row = a(i,:)
            a(i,:) = a(j,:)
            a(j,:) = row

            ! array b
            row = b(i,:)
            b(i,:) = b(j,:)
            b(j,:) = row
        end if
    end do
end subroutine

!-------------------------------------------------------------------------------
! shuffle first 3D ranks of a 4D array and rows of a 2D array in corresponding
! order (Fisher-Yates shuffle)
!-------------------------------------------------------------------------------
! a:         (real(:,:,:,:))
! b:         (real(:,:,:,:))
!-------------------------------------------------------------------------------
! alters ::  images and arr rows shuffled (correspondingly) in-place
!-------------------------------------------------------------------------------
subroutine pair_shuffle_channels_4D(images, arr)
    real              :: images(:,:,:,:), arr(:,:), randn
    real, allocatable :: channel(:,:,:), row(:)
    integer           :: i, j

    ! loop through channels from high to low indices
    do i = size(images, dim=4), 2, -1
        j = i + 1 ! arbitrary assignment to enter loop below

        ! generate random j in range [1,i]
        do while (j > i)
            call random_number(randn) ! range [0, 1)
            ! scale to [0, i) => floor to [0,i-1] => shift to [1,i]
            j = 1 + floor(randn * i)
        end do

        ! swap if random index in different place
        ! (if random j = i, don't need to swap i with i)
        if (i /= j) then
            channel = images(:,:,:,i)
            images(:,:,:,i) = images(:,:,:,j)
            images(:,:,:,j) = channel

            row = arr(i,:)
            arr(i,:) = arr(j,:)
            arr(j,:) = row
        end if
    end do
end subroutine

!-------------------------------------------------------------------------------
! one-hot encode a single variable;
!   *** special format: the type should be real, but values expected to have 0s
!       in decimal place (like an integer variable that was converted to reals);
!       
!   *** values should start with: 0.0, 1.0, 2.0, 3.0, . . .
!-------------------------------------------------------------------------------
! a:        (real(:)) variable
! classes:  (integer) classes encoded in the variable
! res:      (real(:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes one-hot encoding of a
!-------------------------------------------------------------------------------
subroutine one_hot_encode_special(a, classes, res)
    real, intent(in)    :: a(:)
    integer, intent(in) :: classes
    real, allocatable   :: res(:,:)
    integer             :: row, col
    integer(kind=4)     :: val ! to help cast 8-bit double to 4-bit integer

    ! create new array if not correct size
    if (allocated(res)) then
        if (.not. all(shape(res) == [size(a, dim=1), classes])) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(size(a, dim=1), classes))
    end if

    res = 0

    do row = 1, size(a, dim=1)
        val = int(a(row), kind=4) ! cast 8-bit double to 4-bit integer
        col = val + 1             ! convert 0-indexed val to 1-indexed col
        res(row, col) = 1
    end do
end subroutine

!-------------------------------------------------------------------------------
! returns the rows to add (NOT INCLUDING base rows) needed to support a given&
! padding scheme, with a given kernel passing over a given array
!-------------------------------------------------------------------------------
! a_rows:      (integer) base array rows
! kernel_rows: (integer) kernel rows
! stride_rows: (integer) size of kernel moves in y direction
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! returns ::   rows to pad to support given padding scheme
!-------------------------------------------------------------------------------
integer function pad_rows(a_rows, kernel_rows, stride_rows, padding)
    integer, intent(in)      :: a_rows, kernel_rows, stride_rows
    character(*), intent(in) :: padding

    select case(padding)
        case ('same')
            ! pads input so output has same size as input
            pad_rows = (stride_rows - 1) * a_rows - stride_rows + kernel_rows
        case ('full')
            ! pads input so kernel can overlap it by one unit on all sides
            pad_rows = 2 * kernel_rows - 2
        case ('valid')
            ! no padding
            pad_rows = 0
        case default
            print *, '----------------------------------'
            print *, '(net_helper_functions :: pad_rows)'
            print *, 'invalid padding type.'
            print *, 'supported: same, valid, full'
            print *, '----------------------------------'
            stop -1
    end select
end function

!-------------------------------------------------------------------------------
! returns the cols to add (NOT INCLUDING base cols) needed to support a given&
! padding scheme, with a given kernel passing over a given array
!-------------------------------------------------------------------------------
! a_cols:      (integer) base array cols
! kernel_cols: (integer) kernel cols
! stride_cols: (integer) size of kernel moves in x direction
! padding:     (characters) padding type
!-------------------------------------------------------------------------------
! returns ::   cols to pad to support given padding scheme
!-------------------------------------------------------------------------------
integer function pad_cols(a_cols, kernel_cols, stride_cols, padding)
    integer, intent(in)      :: a_cols, kernel_cols, stride_cols
    character(*), intent(in) :: padding

    select case(padding)
        case ('same')
            ! pads input so output has same size as input
            pad_cols = (stride_cols - 1) * a_cols - stride_cols + kernel_cols
        case ('full')
            ! pads input so kernel can overlap it by one unit on all sides
            pad_cols = 2 * kernel_cols - 2
        case ('valid')
            ! no padding
            pad_cols = 0
        case default
            print *, '----------------------------------'
            print *, '(net_helper_functions :: pad_cols)'
            print *, 'invalid padding type.'
            print *, 'supported: same, valid, full'
            print *, '----------------------------------'
            stop -1
    end select
end function

!-------------------------------------------------------------------------------
! pads a 2D array based on a kernel (for cross correlation or convolution)
!-------------------------------------------------------------------------------
! a:           (real(:,:)) base array
! padding:     (characters) padding type
! kernel_dims: (integer(2)) (height, width) of kernel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! res:         (real(:,:)) stores the output
!-------------------------------------------------------------------------------
! alters ::    res becomes result of padding a around its (height, width)
!-------------------------------------------------------------------------------
subroutine pad_2D(a, padding, kernel_dims, stride, res)
    real, intent(in)         :: a(:,:)
    integer, intent(in)      :: kernel_dims(2), stride(2)
    character(*), intent(in) :: padding
    real, allocatable        :: res(:,:)
    integer                  :: a_rows, a_cols, row_pad, col_pad, top_pad, &
                                left_pad

    a_rows = size(a, dim=1)
    a_cols = size(a, dim=2)

    row_pad = pad_rows(a_rows, kernel_dims(1), stride(1), padding)
    col_pad = pad_cols(a_cols, kernel_dims(2), stride(2), padding)

    ! if total pad number odd: top/left get rounded down amount,
    ! (bottom/right get remainder)
    top_pad = row_pad / 2
    left_pad = col_pad / 2

    ! create new array if not correct size
    if (allocated(res)) then
        if (.not. all(shape(res) == [a_rows+row_pad, a_cols+col_pad])) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(row_pad+a_rows, col_pad+a_cols))
    end if

    ! set a in res
    res = 0
    res(top_pad+1:top_pad+a_rows, left_pad+1:left_pad+a_cols) = a
end subroutine

!-------------------------------------------------------------------------------
! pads a 3D array based on a kernel (for cross correlation or convolution)
!-------------------------------------------------------------------------------
! a:           (real(:,:,:)) base array
! padding:     (characters) padding type
! kernel_dims: (integer(2)) (height, width) of kernel
! stride:      (integer(2)) size of kernel moves in (y, x) directions
! res:         (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters ::    res becomes result of padding a around its (height, width)
!-------------------------------------------------------------------------------
subroutine pad_3D(a, padding, kernel_dims, stride, res)
    real, intent(in)         :: a(:,:,:)
    integer, intent(in)      :: kernel_dims(2), stride(2)
    character(*), intent(in) :: padding
    real, allocatable        :: res(:,:,:)
    integer                  :: a_rows, a_cols, a_channels, row_pad, col_pad, &
                                top_pad, left_pad

    a_rows     = size(a, dim=1)
    a_cols     = size(a, dim=2)
    a_channels = size(a, dim=3)

    row_pad = pad_rows(a_rows, kernel_dims(1), stride(1), padding)
    col_pad = pad_cols(a_cols, kernel_dims(2), stride(2), padding)

    ! if total pad number odd: top/left get rounded down amount,
    ! (bottom/right get remainder)
    top_pad = row_pad / 2
    left_pad = col_pad / 2

    ! create new array if not correct size
    if (allocated(res)) then
        if (.not. all(shape(res) == [a_rows+row_pad, a_cols+col_pad, &
                                     a_channels])) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(a_rows+row_pad, a_cols+col_pad, a_channels))
    end if

    ! set a in res
    res = 0
    res(top_pad+1:top_pad+a_rows, left_pad+1:left_pad+a_cols, :) = a
end subroutine

!-------------------------------------------------------------------------------
! expands 3D array so that each row is stride(1) away from the next row,
! and so that each col is stride(2) away from the next col; gaps filled with 0s
!
! as a top-down 2D example, if a = [1 2], and stride = [3, 2],
!                                  [3 4]
!
! res = [1 0 2], becuase it takes 2 moves to go from column 1 to 2,
!       [0 0 0]  and it takes 3 moves to go from row 1 to 2
!       [0 0 0]
!       [3 0 4]
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes result of expanding a with stride
!-------------------------------------------------------------------------------
subroutine expand_with_stride_3D(a, stride, res)
    real, intent(in)    :: a(:,:,:)
    integer, intent(in) :: stride(2)
    real, allocatable   :: res(:,:,:)
    integer             :: rows, cols, channels, res_rows, res_cols, r, c

    if (stride(1) == 1 .and. stride(2) == 1) then
        res = a ! no striding to do
    else
        rows     = size(a, dim=1)
        cols     = size(a, dim=2)
        channels = size(a, dim=3)

        res_rows = rows + (rows - 1) * (stride(1) - 1)
        res_cols = cols + (cols - 1) * (stride(2) - 1)

        ! create new array if not correct size
        if (allocated(res)) then
            if (.not. all(shape(res) == [res_rows, res_cols, channels])) then
                deallocate(res)
            end if
        end if

        if (.not. allocated(res)) then
            allocate(res(res_rows, res_cols, channels))
        end if

        res = 0

        do r = 1, rows
            do c = 1, cols
                ! fill in parts from a
                res((r-1)*stride(1)+1,(c-1)*stride(2)+1,:) = a(r,c,:)
            end do
        end do
    end if
end subroutine

!-------------------------------------------------------------------------------
! calculates the cross-correlation between a 2D array and a 2D kernel
!-------------------------------------------------------------------------------
! a:        (real(:,:)) base array
! padding:  (characters) padding type
! kernel:   (real(:,:)) kernal to pass over a
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes cross-correlation result
!-------------------------------------------------------------------------------
subroutine cross_correlate_2D(a, padding, kernel, stride, res)
    real, intent(in)         :: a(:,:), kernel(:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:)
    real, allocatable        :: padded(:,:)
    integer                  :: a_rows, a_cols, k_rows, k_cols, &
                                padded_rows, padded_cols, &
                                res_rows, res_cols, r, c

    a_rows = size(a, dim=1)
    a_cols = size(a, dim=2)
    k_rows = size(kernel, dim=1)
    k_cols = size(kernel, dim=2)

    call pad_2D(a, padding, [k_rows,k_cols], stride, padded)
    padded_rows = size(padded, dim=1)
    padded_cols = size(padded, dim=2)

    ! dimensions of cross-correlation result
    res_rows = (padded_rows - k_rows) / stride(1) + 1
    res_cols = (padded_cols - k_cols) / stride(2) + 1

    ! create new array if not correct size
    if (allocated(res)) then
        if (.not. all(shape(res) == [res_rows, res_cols])) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(res_rows, res_cols))
    end if

    do r = 1, res_rows
        do c = 1, res_cols
            ! each index in the result is the dot product between the
            ! kernel and a matching-sized section of the padded array
            res(r,c) = sum(padded((r-1)*stride(1)+1:(r-1)*stride(1)+k_rows, &
                                  (c-1)*stride(2)+1:(c-1)*stride(2)+k_cols) * &
                           kernel)
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! calculates the cross-correlation between a 3D array and a 3D kernel
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernel:   (real(:,:,:)) kernal to pass over a
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:)) stores the output (only 1 channel output from dot-prod)
!-------------------------------------------------------------------------------
! alters :: res becomes cross-correlation result
!-------------------------------------------------------------------------------
subroutine cross_correlate_3D(a, padding, kernel, stride, res)
    real, intent(in)         :: a(:,:,:), kernel(:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:), padded(:,:,:)
    integer                  :: a_rows, a_cols, a_channels, &
                                k_rows, k_cols, k_channels, &
                                padded_rows, padded_cols, &
                                res_rows, res_cols, r, c

    a_rows     = size(a, dim=1)
    a_cols     = size(a, dim=2)
    a_channels = size(a, dim=3)
    k_rows     = size(kernel, dim=1)
    k_cols     = size(kernel, dim=2)
    k_channels = size(kernel, dim=3)

    if (a_channels /= k_channels) then
        print *, '--------------------------------------------'
        print *, '(net_helper_functions :: cross_correlate_3D)'
        print *, 'input and kernel channels differ.'
        print *, '--------------------------------------------'
        stop -1
    end if

    call pad_3D(a, padding, [k_rows,k_cols], stride, padded)
    padded_rows = size(padded, dim=1)
    padded_cols = size(padded, dim=2)

    ! dimensions of cross-correlation result
    res_rows = (padded_rows - k_rows) / stride(1) + 1
    res_cols = (padded_cols - k_cols) / stride(2) + 1

    ! create new array if not correct size
    if (allocated(res)) then
        if (.not. all(shape(res) == [res_rows, res_cols])) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(res_rows, res_cols))
    end if

    do r = 1, res_rows
        do c = 1, res_cols
            ! each index in the result is the dot product between the
            ! kernel and a matching-sized section of the padded array
            res(r,c) = sum(padded((r-1)*stride(1)+1:(r-1)*stride(1)+k_rows, &
                                  (c-1)*stride(2)+1:(c-1)*stride(2)+k_cols, &
                                  :) * &
                           kernel)
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! calculates the convolution between a 2D array and a 2D kernel
!-------------------------------------------------------------------------------
! a:        (real(:,:)) base array
! padding:  (characters) padding type
! kernel:   (real(:,:)) kernal to pass over a
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes convolution result
!-------------------------------------------------------------------------------
subroutine convolve_2D(a, padding, kernel, stride, res)
    real, intent(in)         :: a(:,:), kernel(:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:)
    real, allocatable        :: rot_kernel(:,:)
    integer                  :: rows, cols, c

    rows = size(kernel, dim=1)
    cols = size(kernel, dim=2)

    allocate(rot_kernel(rows, cols))

    ! rotate kernel 180 degrees
    do c = 1, cols
        ! res column from left is reverse of column of a from right
        rot_kernel(:,c) = kernel(rows:1:-1, cols+1-c)
    end do

    call cross_correlate_2D(a, padding, rot_kernel, stride, res)
end subroutine

!-------------------------------------------------------------------------------
! calculate the cross-correlation between the 3D input and each 3D kernel (in
! 4D array)
!
! helper for conv_neural_net forward propagation;
! opposite direction of cross_correlate_3D_back and transpose_convolve_3D_back
!-------------------------------------------------------------------------------
! input:    (real(:,:,:)) base array
! padding:  (characters) padding type
! kernel:   (real(:,:,:,:)) channels of 3D kernels
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes 3D cross-correlation forward prop result
!-------------------------------------------------------------------------------
subroutine cross_correlate_3D_forw(input, padding, kernel, stride, res)
    real, intent(in)         :: input(:,:,:), kernel(:,:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:,:), res_channel(:,:)
    integer                  :: kernels, k

    kernels = size(kernel, dim=4)

    ! cross correlate each kernel with input
    do k = 1, kernels
        call cross_correlate_3D(input, padding, kernel(:,:,:,k), stride, &
                                res_channel)

        ! create new array if not correct size
        if (allocated(res)) then
            if (.not. all(shape(res(:,:,1)) == shape(res_channel)) .or. &
                size(res, dim=3) /= kernels) then
                deallocate(res)
            end if
        end if

        if (.not. allocated(res)) then
            allocate(res(size(res_channel, dim=1), size(res_channel, dim=2), &
                         kernels))
        end if

        res(:,:,k) = res_channel
    end do
end subroutine

!-------------------------------------------------------------------------------
! used with result (delta) of cross_correlate_3D_forw (this "reverses" that
! function wrt kernels); calculate the cross-correlation between each delta and
! each corresponding input channel, and sum up the results corresponding to each
! kernel channel that was used in cross_correlate_3D_forw
!
! helper for conv_neural_net backpropagation, to calcualte cost wrt kernels;
! opposite direction of cross_correlate_3D_forw
!-------------------------------------------------------------------------------
! input:    (real(:,:,:)) base array
! padding:  (characters) padding type
! delta:    (real(:,:,:)) channels of deltas (cross_correlate_3D_forw output)
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes 3D cross-correlation back prop result
!-------------------------------------------------------------------------------
subroutine cross_correlate_3D_back(input, padding, delta, stride, res)
    real, intent(in)         :: input(:,:,:), delta(:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:,:,:), res_channel(:,:), exp_delta(:,:,:)
    integer                  :: d_channels, i_channels, d, i

    d_channels = size(delta, dim=3)
    i_channels = size(input, dim=3) ! input channels match kernel channels

    ! account for stride
    call expand_with_stride_3D(delta, stride, exp_delta)

    ! cross-correlate each delta channel with each corresponding
    ! input channel, then sum up the results by kernel channel
    do d = 1, d_channels
        do i = 1, i_channels
            call cross_correlate_2D(input(:,:,i), padding, exp_delta(:,:,d), &
                                    [1,1], res_channel)

            ! create new array if not correct size
            if (allocated(res)) then
                ! deallocate if wrong size
                if (.not. all(shape(res) == [size(res_channel, dim=1), &
                                             size(res_channel, dim=2), &
                                             i_channels, d_channels])) then
                    deallocate(res)
                end if
            end if

            if (.not. allocated(res)) then
                allocate(res(size(res_channel, dim=1), &
                             size(res_channel, dim=2), &
                             i_channels, d_channels))
            end if

            res(:,:,i,d) = res_channel
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! used with result (delta) of cross_correlate_3D_forw (this "reverses" that
! function wrt input); calculate the transpose convolution between each kernel
! and each corresponding delta channel, and sum up the results corresponding to
! each input channel that was used in cross_correlate_3D_forw
!
! helper for conv_neural_net backpropagation, to calcualte cost wrt previous
! layer activations;
! opposite direction of cross_correlate_3D_forw
!-------------------------------------------------------------------------------
! delta:    (real(:,:,:)) channels of deltas (cross_correlate_3D_forw output)
! padding:  (characters) padding type
! kernel:   (real(:,:,:,:)) channels of 3D kernels
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes 3D transpose convolution result
!-------------------------------------------------------------------------------
subroutine transpose_convolve_3D_back(delta, padding, kernel, stride, res)
    real, intent(in)         :: delta(:,:,:), kernel(:,:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:,:), res_channel(:,:), &
                                exp_delta(:,:,:), unpad_channel(:,:)
    integer                  :: d_rows, d_cols, d_channels, k_channels, d, k, &
                                res_row_pad, res_col_pad, top_pad, left_pad

    d_rows     = size(delta, dim=1)
    d_cols     = size(delta, dim=2)
    d_channels = size(delta, dim=3)
    k_channels = size(kernel, dim=3)

    ! account for stride
    call expand_with_stride_3D(delta, stride, exp_delta)

    ! convolve each delta channel with corresponding kernel
    ! channel, then sum up the results by input channel
    do k = 1, k_channels
        do d = 1, d_channels
            call convolve_2D(exp_delta(:,:,d), 'full', kernel(:,:,k,d), [1,1], &
                             res_channel)

            if (padding == 'same') then
                ! res layer results currently includes padding; must remove it
                res_row_pad = size(res_channel, dim=1) - d_rows
                res_col_pad = size(res_channel, dim=2) - d_cols

                ! if odd number pad: top/left get rounded down amount
                top_pad = res_row_pad / 2
                left_pad = res_col_pad / 2

                ! select array (without padding) from res_channel
                unpad_channel = res_channel(top_pad+1:top_pad+d_rows, &
                                            left_pad+1:left_pad+d_cols)
            else if (padding == 'valid') then
                unpad_channel = res_channel ! no padding to remove
            end if

            ! create new array if not correct size
            if (allocated(res)) then
                ! deallocate if wrong size
                if (.not. all(shape(res(:,:,1)) == shape(unpad_channel)) .or. &
                    size(res, dim=3) /= k_channels) then
                    deallocate(res)
                else
                    res = 0 ! allocated, and correct size
                end if
            end if

            if (.not. allocated(res)) then
                allocate(res(size(unpad_channel, dim=1), &
                             size(unpad_channel, dim=2), &
                             k_channels))
                res = 0
            end if

            res(:,:,k) = res(:,:,k) + unpad_channel
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! calculates the max pooling between a 2D array and a 2D kernel
!-------------------------------------------------------------------------------
! a:           (real(:,:)) base array
! padding:     (characters) pool padding type
! kernel_dims: (integer(2)) (height, width) of kernel
! stride:      (real(2)) size of kernel moves in (y, x) directions
! res:         (real(:,:)) stores pooled output
! res_idxs:    (real(:,:,:)) stores chosen max indices
!-------------------------------------------------------------------------------
! alters :: - res becomes max pooling result
!           - res_idxs matches res (rows, cols), where third dimension of size 2
!             is the index (Y, X) of the max value in the padded original array;
!             -1 where max value is in padding, not original array
!-------------------------------------------------------------------------------
subroutine max_pool_2D(a, padding, kernel_dims, stride, res, res_idxs)
    real, intent(in)         :: a(:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: kernel_dims(2), stride(2)
    real, allocatable        :: res(:,:), padded(:,:)
    integer, allocatable     :: res_idxs(:,:,:)
    integer                  :: max_idx(2), a_rows, a_cols, k_rows, k_cols, &
                                padded_rows, padded_cols, top_pad, left_pad, &
                                res_rows, res_cols, r, c

    a_rows = size(a, dim=1)
    a_cols = size(a, dim=2)
    k_rows = kernel_dims(1)
    k_cols = kernel_dims(2)

    call pad_2D(a, padding, [k_rows,k_cols], stride, padded)
    padded_rows = size(padded, dim=1)
    padded_cols = size(padded, dim=2)

    top_pad = (padded_rows - a_rows) / 2
    left_pad = (padded_cols - a_cols) / 2

    ! dimensions of pooling result
    res_rows = (padded_rows - k_rows) / stride(1) + 1
    res_cols = (padded_cols - k_cols) / stride(2) + 1

    ! create new array if not correct size
    if (allocated(res)) then
        if (.not. all(shape(res) == [res_rows, res_cols])) then
            deallocate(res)
        end if
    end if

    if (.not. allocated(res)) then
        allocate(res(res_rows, res_cols))
    end if

    ! (rows, cols) of res_idxs matches up entries to res;
    ! depth of 2 corresponds to (row, col) of idx;
    ! create new array if not correct size
    if (allocated(res_idxs)) then
        if (.not. all(shape(res_idxs) == [res_rows, res_cols, 2])) then
            deallocate(res_idxs)
        end if
    end if

    if (.not. allocated(res_idxs)) then
        allocate(res_idxs(res_rows, res_cols, 2))
    end if

    do r = 1, res_rows
        do c = 1, res_cols
            ! each index in the result of the maximum between the
            ! in a section of the padded array matching the kernel;
            ! max_idx is index WITHIN THE KERNEL; NOT overall array
            max_idx = maxloc(padded((r-1)*stride(1)+1:(r-1)*stride(1)+k_rows, &
                                    (c-1)*stride(2)+1:(c-1)*stride(2)+k_cols))

            ! shift in-kernel max index to index in padded array
            max_idx(1) = max_idx(1) + (r-1)*stride(1)
            max_idx(2) = max_idx(2) + (c-1)*stride(2)

            ! set actual max value and index in padded array
            res_idxs(r,c,:) = max_idx
            res(r,c) = padded(max_idx(1), max_idx(2))

            ! if index is in the pad, not the original array, set (y,x) = -1
            if ((max_idx(1) <= top_pad .or. max_idx(1) > top_pad+a_rows) .or. &
                (max_idx(2) <= left_pad .or. max_idx(2) > left_pad+a_cols)) then
                res_idxs(r,c,:) = [-1,-1]
            end if
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! calculates the max pooling between channels of arrays and a kernel
!-------------------------------------------------------------------------------
! a:           (real(:,:,:)) channels of base arrays
! padding:     (characters) pool padding type
! kernel_dims: (integer(2)) (height, width) of kernel
! stride:      (real(2)) size of kernel moves in (y, x) directions
! res:         (real(:,:,:)) stores pooled output; overwritten
! res_idxs:    (real(:,:,:,:)) stores chosen max indices; overwritten
!-------------------------------------------------------------------------------
! alters ::    *res becomes 3D max pooling result
!              *res_idxs becomes 4D array of max indices; 4th dimension is the
!               channel, first three dimensions match those returned by res_idxs
!               from max_pool_2D (see description above);
!               -1 where max value in padding, not original array
!-------------------------------------------------------------------------------
subroutine max_pool_3D(a, padding, kernel_dims, stride, res, res_idxs)
    real, intent(in)         :: a(:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: kernel_dims(2), stride(2)
    real, allocatable        :: res_channel(:,:), res(:,:,:)
    integer, allocatable     :: channel_idxs(:,:,:), res_idxs(:,:,:,:)
    integer                  :: k, a_channels

    a_channels = size(a, dim=3)

    do k = 1, a_channels
        call max_pool_2D(a(:,:,k), padding, kernel_dims, stride, res_channel, &
                         channel_idxs)

        ! create new array if not correct size
        if (allocated(res)) then
            if (.not. all(shape(res) == [size(res_channel, dim=1), &
                                         size(res_channel, dim=2), &
                                         a_channels])) then
                deallocate(res)
            end if
        end if

        if (.not. allocated(res)) then
            allocate(res(size(res_channel, dim=1), size(res_channel, dim=2), &
                         a_channels))
        end if

        ! create new array if not correct size
        if (allocated(res_idxs)) then
            if (.not. all(shape(res_idxs) == [size(channel_idxs, dim=1), &
                                              size(channel_idxs, dim=2), &
                                              2, a_channels])) then
                deallocate(res_idxs)
            end if
        end if

        if (.not. allocated(res_idxs)) then
            allocate(res_idxs(size(channel_idxs, dim=1), &
                              size(channel_idxs, dim=2), &
                              2, a_channels))
        end if

        ! set max pool and max indices for channel
        res(:,:,k) = res_channel
        res_idxs(:,:,:,k) = channel_idxs
    end do
end subroutine
end module
