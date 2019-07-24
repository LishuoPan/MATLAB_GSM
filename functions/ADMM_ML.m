function [AlphaReturn, AugObjEval, OriObjEval, Gap, SSubIterList, TimeList, MSEList] = ADMM_ML(xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,K,options)
%ADMM_ML ADMM framework for MLK Optimization
%   Input class support:
%       ytrain: training y, column vector;
%       U: cell of Kernels;
%       options: strut(rho, MAX_iter, nv, iniAlpha).
%   Output:
%       alpha: column vector
%   dependency:
%       C_matrix.m
%       S_gradient.m

% start clock
tic
    % define constants
    Q = numel(K);
    n = length(ytrain);
    I_Matrix = eye(n);
    % convergence criteria
    AugObjEval = zeros(options.MAX_iter+1,1);
    OriObjEval = zeros(options.MAX_iter+1,1);
    Gap = zeros(options.MAX_iter+1,1);
    % Record important information
    SSubIterList = zeros(options.MAX_iter,1);
    TimeList = zeros(options.MAX_iter+1,1);
    MSEList = zeros(options.MAX_iter+1,1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initialization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Alpha_k = options.iniAlpha;
    L_k = I_Matrix;
    C_k = C_matrix(Alpha_k, K, options.nv, I_Matrix);
    S_k = inv(C_k);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % START ADMM ITERATIONS
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % display info
    disp(['Solver: ADMM      ','rho = ',num2str(options.rho), ...
        '  dual rho = ',num2str(options.rho_dual),'  inner loop:',int2str(options.MaxIL)]);
    disp('It.    AugObj        OriObj        MSE           norm2diff_alpha    time    L')

    for i= 1:options.MAX_iter
        AugObjEval(i) = AugObj(ytrain, S_k, L_k, C_k, options.rho);
        OriObjEval(i) = ML_obj(C_k, ytrain);
        Gap(i) = norm(S_k*C_k - I_Matrix,'fro');
        % The following 3 lines are for record use. Should not be in the
        % final version of the code.
        TimeList(i) = toc;
        [pMean, ~] = prediction(xtrain,xtest,ytrain,nTest,Alpha_k,varEst,freq,var,K);
        MSEList(i) = mean((pMean-ytest(1:nTest)).^2);
        %%%%%%%%%%%%%%%%%%%%
        % S update
        %%%%%%%%%%%%%%%%%%%%
        % gradient descent update
        [S_k,SSubIter] = SUpdate(ytrain, S_k, L_k, C_k, options.rho, options.MaxIL);
        SSubIterList(i) = SSubIter;
        % display S matrix Non-PD info
        [~,PD] = chol(S_k);
        if PD ~= 0
            disp('Warning: S matrix is Non-PD, this may lead to failure');
        end

        %%%%%%%%%%%%%%%%%%%%
        % alpha update
        %%%%%%%%%%%%%%%%%%%%
        LastAlpha = Alpha_k;
        for ii=1:Q
            % pre-calculate O(n^3)
            ske = S_k*K{ii};
            sc = S_k*C_k;
            OldAlphaii = Alpha_k(ii);
            Alpha_k(ii) = max(0,alpha_update(ske, sc, options.rho, L_k, OldAlphaii));
            % update new C_k
            C_k = C_k - OldAlphaii*K{ii} + Alpha_k(ii)*K{ii};
        end
        diff_alpha = norm(LastAlpha-Alpha_k);
        % stopping criteria
        if diff_alpha < 0.001
            disp('Optimal Alpha Found.');
            AlphaReturn = Alpha_k;
            AugObjEval(i+1) = AugObj(ytrain, S_k, L_k, C_k, options.rho);AugObjEval = AugObjEval(1:i+1);
            OriObjEval(i+1) = ML_obj(C_k, ytrain);OriObjEval = OriObjEval(1:i+1);
            Gap(i+1) = norm(S_k*C_k - I_Matrix,'fro');Gap = Gap(1:i+1);
            return
        end
        
        %%%%%%%%%%%%%%%%%%%%
        % L update
        %%%%%%%%%%%%%%%%%%%%
        % Close form update L. Use a smaller dual coefficient
        L_k = L_k + options.rho_dual*(S_k*C_k - I_Matrix);
        
        %%%%%%%%%%%%%%%%%%%%
        % Print Report
        %%%%%%%%%%%%%%%%%%%%
        % report every 100 iterations.
        if rem(i,100)==0
            % prediction & report the MSE
            [pMean, ~] = prediction(xtrain,xtest,ytrain,nTest,Alpha_k,varEst,freq,var,K);
            MSE = mean((pMean-ytest(1:nTest)).^2);
            % plot
%             figName = './fig/ADMM_Temp';
%             plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName)

            OriObjPrint = ML_obj(C_k, ytrain);
            AugObjPrint = AugObj(ytrain, S_k, L_k, C_k, options.rho);
            disp([sprintf('%-4d',i),'   ', ...
                  sprintf('%0.4e',AugObjPrint),'    ', ...
                  sprintf('%0.4e',OriObjPrint),'    ', ...
                  sprintf('%0.4e',MSE), '    ', ...
                  sprintf('%0.4e',diff_alpha), '         ', ...
                  sprintf('%-.2f',toc), '    ', ...
                  sprintf('%0.4e',norm(L_k,'fro')^2)]);
        end
        % end of Print
        %%%%%%%%%%%%%%%%%%%%
    end
    % record the convergence criteria
    AugObjEval(options.MAX_iter+1) = AugObj(ytrain, S_k, L_k, C_k, options.rho);
    OriObjEval(options.MAX_iter+1) = ML_obj(C_k, ytrain);
    Gap(options.MAX_iter+1) = norm(S_k*C_k - I_Matrix,'fro');
    % 3 lines for the record use. Should not be in the final version
    TimeList(options.MAX_iter+1) = toc;
    [pMean, ~] = prediction(xtrain,xtest,ytrain,nTest,Alpha_k,varEst,freq,var,K);
    MSEList(options.MAX_iter+1) = mean((pMean-ytest(1:nTest)).^2);
    % Max It. Reached. Module Return Alpha
    disp('Exceed Max Iterations.')
    AlphaReturn = Alpha_k;
end





