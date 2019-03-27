function [iniAlpha, status] = alphaIniFromPeriodogram(yData,nGrids,mus,sigmasq)
%
%Inputs: 
%
%      1.yData: training output
%      2.nGrids: number of grids/frequecy candidates, also an indicator of
%                model complexity
%      3.mus: a vector of frequency candidates
%      4.sigma: a small scalar variance same for all frequencies in the
%             second configuration of GSM kernel 
%
%Output:
%      1. iniAlpha: a hopefully "good" initial guess. 
%      2. status: an indicator of goodness of this initial guess. "fail"
%                 means unrealiable initial guess.

%Evaluate the periodogram at discrete frequency points
%nfft = nGrids*2;
[pyy,w] = pwelch(yData,[],[],mus*2*pi,'onesided');

%Plot the periodogram
plot(w, 10*log(pyy));

%Construct the "data matrix" Psi 
Psi = zeros(nGrids, nGrids);

for ii=1:nGrids
    
    Psi(ii,:) = normpdf(mus(ii), mus, sqrt(sigmasq));
    
end

%Solve the L-1 norm regularization problem using Prof. Stephen Boyd's
%routine. 

%Algorithm setup
lambda = 1000;      % regularization parameter (can be tuned)
rel_tol = 10;     % relative target duality gap (can be tuned)

[iniAlpha,status]=l1_ls(Psi,pyy',lambda,rel_tol);

iniAlpha = max(iniAlpha, 0);

end

