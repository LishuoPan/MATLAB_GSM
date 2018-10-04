function diff_mat = diff_mat(x_train)
%DIFF_MAT difference matrix
%   diff_mat return the difference matrix based on the x_train
%
%   Class support for the input x_train:
%       float: double; column vector
%       training data x

diff_mat = x_train - x_train';

end