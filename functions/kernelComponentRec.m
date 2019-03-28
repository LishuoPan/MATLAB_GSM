function K = kernelComponentRec(freq_grid, var_grid, x, y)
%Given the lists of hyperparameters and the difference matrix, compute all
%the sub kernel matrices and store in a cell.

Q = length(freq_grid);
subKernels = cell(Q,1);

% compute the difference matrix based on x_train
diffMat = diff_mat(x, y);

% constructing the Kernel cells
for k=1:Q
    freqPara = freq_grid(k);
    varPara = var_grid(k);
    subKernels{k} = sinc(pi*varPara.*diffMat).*cos(2*pi*freqPara*diffMat);
end
K = subKernels;

end