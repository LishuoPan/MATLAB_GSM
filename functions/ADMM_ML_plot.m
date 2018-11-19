function alpha = ADMM_ML_plot(xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,U,options)
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
    Q = numel(U);
    d = length(ytrain);
    eyeM = eye(d);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initialization
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % initialize the hyperparameter
    alpha_k = options.iniAlpha;
    L_k = eye(d);
    % the first c_k put into the process
    c_k = C_matrix(alpha_k, U, options.nv, eyeM);
    S_k = inv(c_k);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % START ADMM ITERATION UPDATE
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % display info
    disp(['The hyperparameter rho = ',sprintf('%0.20e',options.rho)]);


    for i= 1:options.MAX_iter

        %%%%%%%%%%%%%%%%%%%%
        % S update
        %%%%%%%%%%%%%%%%%%%%
        % gradient descent update
        % method for gradient descent:
%         gradient_method = 1;
        % 0 for original(include inv(S))
        % 1 for approximate(c_k replace inv(S))
        
        step = 1e-16;
        for ii=1:1000
            gradient = S_gradient(ytrain, S_k, L_k, c_k, options.rho, options.gradient_method);
            S_k = S_k - step * gradient;
        end

        % display when S_k is not PD
        [~,PD] = chol(S_k);
        if PD ~= 0
            disp(['if the S is PD(0 is true): ',int2str(PD)]);
        end

        %%%%%%%%%%%%%%%%%%%%
        % alpha update
        %%%%%%%%%%%%%%%%%%%%
        for ii=1:Q
            % pre-calculate O(n^3)
            ske = S_k*U{ii};
            sc = S_k*c_k;
            old_alpha = alpha_k(ii);
            alpha_k(ii) = max(0,alpha_update(ske, sc, options.rho, L_k, old_alpha));
            % update new c_k
            c_k = c_k - old_alpha*U{ii} + alpha_k(ii)*U{ii};
        end
        if rem(i,50)==0
            % iter display
            disp(['Iteration ',int2str(i),' is running...']);
            % prediction (test phase)
            [pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha_k,varEst,freq,var,U);
            % [pMean, pVar] = prediction(xtest,nTest,xtrain,ytrain,nTrain,K,alpha,Q,nv,freq,var);
            MSE = mean((pMean-ytest(1:nTest)).^2)
            % record time
            toc
            % plot phase
            figName = './fig/ADMM_Temp';
            plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName)
        end
        %%%%%%%%%%%%%%%%%%%%
        % L update
        %%%%%%%%%%%%%%%%%%%%

        % c_k is ready

        L_k = L_k + options.rho*(S_k*c_k - eyeM);
        if rem(i,50)==0
            % display the fnorm of BIG LAMBDA as a reference of convergence
            disp(['BIG LAMBDA: ',int2str(norm(L_k,'fro')^2)])
        end

    end

    % module return final alpha
    alpha = alpha_k;
end





