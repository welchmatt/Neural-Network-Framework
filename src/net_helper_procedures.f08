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
! procedures with 2D array input require variables-as-columns form;
! procedures with 4D array input require (rows, columns, channels, batches) form
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
subroutine out_activfunc_2D(z, out_activ, res)
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
real function mse_func_2D(preds, labels)
    real, intent(in) :: preds(:,:), labels(:,:)
    mse_func_2D = sum((preds - labels) ** 2) / (2 * size(preds, dim=1))
end function

!-------------------------------------------------------------------------------
! mean square error between predictions and true labels
!-------------------------------------------------------------------------------
! preds:      (real(:,:,:,:)) predictions
! labels:     (real(:,:,:,:)) targets we want to predict
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function mse_func_4D(preds, targets)
    real, intent(in) :: preds(:,:,:,:), targets(:,:,:,:)
    mse_func_4D = sum((preds - targets) ** 2) / (2 * size(preds, dim=1))
end function

!-------------------------------------------------------------------------------
! categorical cross entropy between predictions and true labels
!-------------------------------------------------------------------------------
! preds:      (real(:,:)) predictions
! labels:     (real(:,:)) targets we want to predict
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function cross_entropy_func_2D(preds, labels)
    real, intent(in) :: preds(:,:), labels(:,:)
    cross_entropy_func_2D = -sum(labels * log(preds)) / size(preds, dim=1)
end function

!-------------------------------------------------------------------------------
! calculate the accuracy between predictions and one-hot label rows
!-------------------------------------------------------------------------------
! preds:      (real(:,:)) predictions
! labels:     (real(:,:)) ONE-HOT ENCODED targets we want to predict
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function one_hot_accuracy_2D(preds, labels)
    real, intent(in) :: preds(:,:), labels(:,:)
    real             :: correct

    ! correct where strongest predictions match one-hot labels
    correct = count(maxloc(preds, dim=2) == maxloc(labels, dim=2))
    one_hot_accuracy_2D = correct / size(preds, dim=1)
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
real function lossfunc_2D(preds, labels, loss)
    real, intent(in)         :: preds(:,:), labels(:,:)
    character(*), intent(in) :: loss

    select case(loss)
        case ('mse')
            lossfunc_2D = mse_func_2D(preds, labels)
        case ('cross_entropy')
            lossfunc_2D = cross_entropy_func_2D(preds, labels)
        case default
            print *, '----------------------------------'
            print *, '(net_helper_functions :: lossfunc)'
            print *, 'invalid loss function.'
            print *, 'supported: mse, cross_entropy'
            print *, '----------------------------------'
            stop -1
    end select
end function

!-------------------------------------------------------------------------------
! wrapper for loss functions
!-------------------------------------------------------------------------------
! preds:      (real(:,:,:,:)) predictions
! targets:    (real(:,:,:,:)) targets we want to predict
! loss:       (characters) loss function
!-------------------------------------------------------------------------------
! returns ::  (real)
!-------------------------------------------------------------------------------
real function lossfunc_4D(preds, targets, loss)
    real, intent(in)         :: preds(:,:,:,:), targets(:,:,:,:)
    character(*), intent(in) :: loss

    select case(loss)
        case ('mse')
            lossfunc_4D = mse_func_4D(preds, targets)
        case default
            print *, '----------------------------------'
            print *, '(net_helper_functions :: lossfunc)'
            print *, 'invalid loss function.'
            print *, 'supported: mse'
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
! shuffle each batch element in images and each row in arr in corresponding
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
! returns the amount to add (NOT INCLUDING base dims) needed to support a given
! padding scheme, with a given kernel passing over a given array
!-------------------------------------------------------------------------------
! a_dim:      (integer) base array dimension
! kernel_dim: (integer) kernel dimension
! stride_dim: (integer) size of kernel moves in y direction
! padding:    (characters) padding type
!-------------------------------------------------------------------------------
! returns ::  amount to pad to support given padding scheme
!-------------------------------------------------------------------------------
integer function pad_calc(a_dim, kernel_dim, stride_dim, padding)
    integer, intent(in)      :: a_dim, kernel_dim, stride_dim
    character(*), intent(in) :: padding

    select case(padding)
        case ('same')
            ! pads input so output has same size as input
            pad_calc = (stride_dim - 1) * a_dim - stride_dim + kernel_dim
        case ('full')
            ! pads input so kernel can overlap it by one unit on all sides
            pad_calc = 2 * kernel_dim - 2
        case ('valid')
            ! no padding
            pad_calc = 0
        case default
            print *, '----------------------------------'
            print *, '(net_helper_functions :: pad_calc)'
            print *, 'invalid padding type.'
            print *, 'supported: same, valid, full'
            print *, '----------------------------------'
            stop -1
    end select
end function

!-------------------------------------------------------------------------------
! returns the size of a dimension after being passed over with 'valid'
! cross correlation; typically used by first calculating the pad and including
! that total amaount as the a_dim
!-------------------------------------------------------------------------------
! a_dim:      (integer) base array dimension
! kernel_dim: (integer) kernel dimension
! stride_dim: (integer) size of kernel moves in y direction
!-------------------------------------------------------------------------------
! returns ::  resulting dim size
!-------------------------------------------------------------------------------
integer function res_calc(a_dim, kernel_dim, stride_dim)
    integer, intent(in) :: a_dim, kernel_dim, stride_dim

    res_calc = (a_dim - kernel_dim) / stride_dim + 1
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

    row_pad = pad_calc(a_rows, kernel_dims(1), stride(1), padding)
    col_pad = pad_calc(a_cols, kernel_dims(2), stride(2), padding)

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

    row_pad = pad_calc(a_rows, kernel_dims(1), stride(1), padding)
    col_pad = pad_calc(a_cols, kernel_dims(2), stride(2), padding)

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
    res_rows = res_calc(padded_rows, k_rows, stride(1))
    res_cols = res_calc(padded_cols, k_cols, stride(2))

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
    res_rows = res_calc(padded_rows, k_rows, stride(1))
    res_cols = res_calc(padded_cols, k_cols, stride(2))

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
! calculates the convolution between a 3D array and a 3D kernel
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernel:   (real(:,:,:)) kernal to pass over a
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:)) stores the output (only 1 channel output from dot-prod)
!-------------------------------------------------------------------------------
! alters :: res becomes convolution result
!-------------------------------------------------------------------------------
subroutine convolve_3D(a, padding, kernel, stride, res)
    real, intent(in)         :: a(:,:,:), kernel(:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:), rot_kernel(:,:,:)
    integer                  :: rows, cols, chans, c

    rows  = size(kernel, dim=1)
    cols  = size(kernel, dim=2)
    chans = size(kernel, dim=3)

    allocate(rot_kernel(rows, cols, chans))

    ! rotate kernel 180 degrees
    do c = 1, cols
        ! res column from left is reverse of column of a from right (all depth)
        rot_kernel(:,c,:) = kernel(rows:1:-1, cols+1-c, :)
    end do

    call cross_correlate_3D(a, padding, rot_kernel, stride, res)
end subroutine

!-------------------------------------------------------------------------------
! cross-correlate each 3D kernel (in 4D array) with a 3D input; return a 3D
! array where each layer corresponds to a cross-correlation result by kernel;
!
! used in forward prop for convolutional layers
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernels:  (real(:,:,:,:)) channels of 3D kernels
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes 3D stack of cross-correlations (kernel count depth)
!-------------------------------------------------------------------------------
subroutine cross_correlate_3D_kernels(a, padding, kernels, stride, res)
    real, intent(in)         :: a(:,:,:), kernels(:,:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:,:), res_channel(:,:)
    integer                  :: k_count, k

    k_count = size(kernels, dim=4)

    ! cross correlate each kernel with input
    do k = 1, k_count
        call cross_correlate_3D(a, padding, kernels(:,:,:,k), stride, &
                                res_channel)

        ! create new array if not correct size
        if (allocated(res)) then
            if (.not. all(shape(res(:,:,1)) == shape(res_channel)) .or. &
                size(res, dim=3) /= k_count) then
                deallocate(res)
            end if
        end if

        if (.not. allocated(res)) then
            allocate(res(size(res_channel, dim=1), size(res_channel, dim=2), &
                         k_count))
        end if

        res(:,:,k) = res_channel
    end do
end subroutine

!-------------------------------------------------------------------------------
! transpose-convolve each 3D kernel (in 4D array) with a 3D input; return a 3D
! array where each layer corresponds to a transpose-convolution result by kernel
!
! used in forward prop for deconvolutional layers
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernels:  (real(:,:,:,:)) channels of 3D kernels
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes 3D stack of transpose-convolutions (kernel count depth)
!-------------------------------------------------------------------------------
subroutine transpose_convolve_3D_kernels(a, padding, kernels, stride, res)
    real, intent(in)         :: a(:,:,:), kernels(:,:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:,:), res_channel(:,:)
    integer                  :: k_count, k

    if (padding /= 'full') then
        print *, '-------------------------------------------------------'
        print *, '(net_helper_functions :: transpose_convolve_3D_kernels)'
        print *, 'must use full padding.'
        print *, '-------------------------------------------------------'
        stop -1
    end if

    k_count = size(kernels, dim=4)

    ! convolve each kernel with input
    do k = 1, k_count
        call convolve_3D(a, padding, kernels(:,:,:,k), stride, res_channel)

        ! create new array if not correct size
        if (allocated(res)) then
            if (.not. all(shape(res(:,:,1)) == shape(res_channel)) .or. &
                size(res, dim=3) /= k_count) then
                deallocate(res)
            end if
        end if

        if (.not. allocated(res)) then
            allocate(res(size(res_channel, dim=1), size(res_channel, dim=2), &
                         k_count))
        end if

        res(:,:,k) = res_channel
    end do
end subroutine

!-------------------------------------------------------------------------------
! cross-correlate all permutations between channels in base array and kernel
! array; result is 4D array, where first 2D are cross-correlation results,
! 3D corresponds to base array channels, 4D corresponds to kernel channels;
! we group the cross-correlation results by kernel channel;
! user can choose to stride either (but not both) arrays with stride;
!
! used in backprop wrt kernels for convolutional layers
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernel:   (real(:,:,:)) kernel array
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:,:)) stores the output
!
! exp_side: (optional - characters) stride expand 'left' (a) or 'right' (kernel)
!-------------------------------------------------------------------------------
! alters :: res becomes cross-correlation permutation result, grouped by kernel
!-------------------------------------------------------------------------------
subroutine cross_correlate_3D_perms_group_kernel(a, padding, kernel, stride, &
                                                 exp_side, res)
    real, intent(in)                   :: a(:,:,:), kernel(:,:,:)
    character(*), intent(in)           :: padding
    integer, intent(in)                :: stride(2)
    character(*), intent(in), optional :: exp_side
    real, allocatable                  :: res(:,:,:,:), res_channel(:,:), &
                                          exp_a(:,:,:), exp_k(:,:,:)
    integer                            :: a_channels, k_channels, a_i, k_i

    a_channels = size(a, dim=3)
    k_channels = size(kernel, dim=3)

    ! handle any stride expansions
    if (present(exp_side)) then
        if (exp_side == 'left') then
            call expand_with_stride_3D(a, stride, exp_a) ! a = left
            exp_k = kernel                               ! default
        else if (exp_side == 'right') then
            call expand_with_stride_3D(kernel, stride, exp_k) ! kernel = right
            exp_a = a                                         ! default
        end if
    else
        ! no expanding
        exp_a = a
        exp_k = kernel
    end if

    ! cross-correlate each corresponding channel pair, grouped by b
    do k_i = 1, k_channels
        do a_i = 1, a_channels
            call cross_correlate_2D(exp_a(:,:,a_i), padding, exp_k(:,:,k_i), &
                                    [1,1], res_channel)

            ! create new array if not correct size
            if (allocated(res)) then
                ! deallocate if wrong size
                if (.not. all(shape(res) == [size(res_channel, dim=1), &
                                             size(res_channel, dim=2), &
                                             a_channels, k_channels])) then
                    deallocate(res)
                end if
            end if

            if (.not. allocated(res)) then
                allocate(res(size(res_channel, dim=1), &
                             size(res_channel, dim=2), &
                             a_channels, k_channels))
            end if

            ! in each group of k_i, we store cross-correlate pairs of all a_i
            res(:,:,a_i,k_i) = res_channel
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! cross-correlate all permutations between channels in base array and kernel
! array; result is 4D array, where first 2D are cross-correlation results,
! 3D corresponds to kernel channels, 4D corresponds to base array channels;
! we group the cross-correlation results by base array channel;
! user can choose to stride either (but not both) arrays with stride;
!
! used in backprop wrt kernels for deconvolutional layers
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernel:   (real(:,:,:)) kernel array
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:,:)) stores the output
!
! exp_side: (optional - characters) stride expand 'left' (a) or 'right' (kernel)
!-------------------------------------------------------------------------------
! alters :: res becomes cross-correlation permutation result, grouped by base
!-------------------------------------------------------------------------------
subroutine cross_correlate_3D_perms_group_base(a, padding, kernel, stride, &
                                               exp_side, res)
    real, intent(in)                   :: a(:,:,:), kernel(:,:,:)
    character(*), intent(in)           :: padding
    integer, intent(in)                :: stride(2)
    character(*), intent(in), optional :: exp_side
    real, allocatable                  :: res(:,:,:,:), res_channel(:,:), &
                                          exp_a(:,:,:), exp_k(:,:,:)
    integer                            :: a_channels, k_channels, a_i, k_i

    if (padding /= 'full') then
        print *, '-------------------------------------------------------'
        print *, '(net_helper_functions :: transpose_convolve_3D_kernels)'
        print *, 'must use full padding.'
        print *, '-------------------------------------------------------'
        stop -1
    end if

    a_channels = size(a, dim=3)
    k_channels = size(kernel, dim=3)

    ! handle any stride expansions
    if (present(exp_side)) then
        if (exp_side == 'left') then
            call expand_with_stride_3D(a, stride, exp_a) ! a = left
            exp_k = kernel                               ! default
        else if (exp_side == 'right') then
            call expand_with_stride_3D(kernel, stride, exp_k) ! kernel = right
            exp_a = a                                         ! default
        end if
    else
        ! no expanding
        exp_a = a
        exp_k = kernel
    end if

    ! cross-correlate each corresponding channel pair, grouped by b
    do a_i = 1, a_channels
        do k_i = 1, k_channels
            ! assumed for deconvolution; use valid rather than full padding
            call cross_correlate_2D(exp_a(:,:,a_i), 'valid', exp_k(:,:,k_i), &
                                    [1,1], res_channel)

            ! create new array if not correct size
            if (allocated(res)) then
                ! deallocate if wrong size
                if (.not. all(shape(res) == [size(res_channel, dim=1), &
                                             size(res_channel, dim=2), &
                                             k_channels, a_channels])) then
                    deallocate(res)
                end if
            end if

            if (.not. allocated(res)) then
                allocate(res(size(res_channel, dim=1), &
                             size(res_channel, dim=2), &
                             k_channels, a_channels))
            end if

            ! in each group of a_i, we store cross-correlate pairs of all k_i
            res(:,:,k_i,a_i) = res_channel
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! transpose-convolve all permutations between channels in base array and kernels
! in kernel array; result is 3D array, where first 2D are transpose-convolution
! results, and 3D sums up the transpose-convolution by kernel channel (3D of
! kernels, where 3D for channels in each kernel, and 4D is different kernels);
!
! used in backprop wrt inputs for convolutional layers, so we must handle
! undoing padding to match input dimensions in this case
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernels:  (real(:,:,:,:)) array of 3D kernels
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes transpose-convolution permutation result, sum by kernel
!-------------------------------------------------------------------------------
subroutine transpose_convolve_3D_perms_sum_kernel(a, padding, kernels, &
                                                  stride, res)
    real, intent(in)         :: a(:,:,:), kernels(:,:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:,:), res_channel(:,:), exp_a(:,:,:), &
                                unpad_channel(:,:)
    integer                  :: a_rows, a_cols, a_channels, k_channels, &
                                res_row_pad, res_col_pad, top_pad, left_pad, &
                                a_i, k_i

    a_rows     = size(a, dim=1)
    a_cols     = size(a, dim=2)
    a_channels = size(a, dim=3)
    k_channels = size(kernels, dim=3)

    ! account for stride
    call expand_with_stride_3D(a, stride, exp_a)

    ! convolve each delta channel with corresponding kernel
    ! channel, then sum up the results by input channel
    do k_i = 1, k_channels
        do a_i = 1, a_channels
            call convolve_2D(exp_a(:,:,a_i), 'full', kernels(:,:,k_i,a_i), &
                             [1,1], res_channel)

            if (padding == 'same') then
                ! res layer results currently includes padding; must remove it
                res_row_pad = size(res_channel, dim=1) - a_rows
                res_col_pad = size(res_channel, dim=2) - a_cols

                ! if odd number pad: top/left get rounded down amount
                top_pad = res_row_pad / 2
                left_pad = res_col_pad / 2

                ! select array (without padding) from res_channel
                unpad_channel = res_channel(top_pad+1:top_pad+a_rows, &
                                            left_pad+1:left_pad+a_cols)
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

            ! sum up results by kernel channel
            res(:,:,k_i) = res(:,:,k_i) + unpad_channel
        end do
    end do
end subroutine

!-------------------------------------------------------------------------------
! cross-correlate all permutations between channels in base array and kernels
! in kernel array; result is 3D array, where first 2D are cross-correlation
! results, and 3D sums up the cross-correlation by kernel channel (3D of
! kernels, where 3D for channels in each kernel, and 4D is different kernels);
!
! used in backprop wrt inputs for deconvolutional layers, so we do not need to
! undo any padding, because deconvolution is assumed have no extra padding
!-------------------------------------------------------------------------------
! a:        (real(:,:,:)) base array
! padding:  (characters) padding type
! kernels:  (real(:,:,:,:)) array of 3D kernels
! stride:   (integer(2)) size of kernel moves in (y, x) directions
! res:      (real(:,:,:)) stores the output
!-------------------------------------------------------------------------------
! alters :: res becomes cross-correlation permutation result, sum by kernel
!-------------------------------------------------------------------------------
subroutine cross_correlate_3D_perms_sum_kernel(a, padding, kernels, stride, res)
    real, intent(in)         :: a(:,:,:), kernels(:,:,:,:)
    character(*), intent(in) :: padding
    integer, intent(in)      :: stride(2)
    real, allocatable        :: res(:,:,:), res_channel(:,:), exp_a(:,:,:)
    integer                  :: a_rows, a_cols, a_channels, k_channels, a_i, k_i

    if (padding /= 'full') then
        print *, '-------------------------------------------------------'
        print *, '(net_helper_functions :: transpose_convolve_3D_kernels)'
        print *, 'must use full padding.'
        print *, '-------------------------------------------------------'
        stop -1
    end if

    a_rows     = size(a, dim=1)
    a_cols     = size(a, dim=2)
    a_channels = size(a, dim=3)
    k_channels = size(kernels, dim=3)

    ! account for stride
    call expand_with_stride_3D(a, stride, exp_a)

    ! convolve each delta channel with corresponding kernel
    ! channel, then sum up the results by input channel
    do k_i = 1, k_channels
        do a_i = 1, a_channels
            ! assumed for deconvolution; use valid rather than full padding
            call cross_correlate_2D(exp_a(:,:,a_i), 'valid', &
                                    kernels(:,:,k_i,a_i), [1,1], res_channel)

            ! create new array if not correct size
            if (allocated(res)) then
                ! deallocate if wrong size
                if (.not. all(shape(res(:,:,1)) == shape(res_channel)) .or. &
                    size(res, dim=3) /= k_channels) then
                    deallocate(res)
                else
                    res = 0 ! allocated, and correct size
                end if
            end if

            if (.not. allocated(res)) then
                allocate(res(size(res_channel, dim=1), &
                             size(res_channel, dim=2), &
                             k_channels))
                res = 0
            end if

            ! sum up results by kernel channel
            res(:,:,k_i) = res(:,:,k_i) + res_channel
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
    res_rows = res_calc(padded_rows, k_rows, stride(1))
    res_cols = res_calc(padded_cols, k_cols, stride(2))

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