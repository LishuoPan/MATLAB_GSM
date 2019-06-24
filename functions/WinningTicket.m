function musk = WinningTicket(PruneIters, PruneRate, SubTrainIters, ...
                              xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,K,options_ADMM);
% The function is to find the winning tickets before the actual training.
% By iterative pruning Pre-training before the 

% Initial Musk Index 1:Q, here musk means the index retained after the pruning
Q = numel(K);
musk = 1:Q;
% Save original iniAlpha in cache
iniAlphaOrig = options_ADMM.iniAlpha;
% Lock the MAX_iter to j in each sub-training
options_ADMM.MAX_iter = SubTrainIters;

% Iterative Prune Process
for i = 1:PruneIters
    % Pruned Alpha in current iteration
    options_ADMM.iniAlpha = iniAlphaOrig(musk);
    % Pruned K and corresponding Freq, Var in current iteration
    SubK = K(musk);
    SubFreq = freq(musk);
    SubVar = var(musk);
    % Iterative Alpha is calculated by
    IterAlpha = ADMM_ML(xtrain,xtest,ytrain,ytest,nTest,varEst,SubFreq,SubVar,SubK,options_ADMM);
    PrunedAmount = floor(PruneRate*length(IterAlpha));
    [~, idx] = sort(IterAlpha);
    % Prune musk
    musk(idx(1:PrunedAmount)) = [];
end

end