function [pMean, pVar] = prediction(xtest,nTest,xtrain,ytrain,nTrain,K,alpha,Q,nv,freq_grid,var_grid)
%This function makes prediction about the test data and evaluate the
%prediction by calculating mean square of error.

kernelFound = zeros(nTrain,nTrain);
for k=1:Q
    kernelFound = kernelFound + alpha(k)*(K{k});
end
invCovMat = pinv(kernelFound + nv*eye(nTrain));

c_subKernel = kernelComponent(freq_grid,var_grid,xtest,xtrain);
c_Kernel = zeros(nTest,nTrain);
for k=1:Q
    c_Kernel = c_Kernel + alpha(k)*(c_subKernel{k});
end
mean = c_Kernel * invCovMat * ytrain;
var=zeros(nTest,1);
s_alpha = sum(alpha);
for i=1:nTest
    kernel_row=c_Kernel(i,:);
    var(i) = s_alpha + nv - kernel_row*invCovMat*(kernel_row.');
end
pMean = mean;
pVar = var;
end