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
    disp(['rho = ',num2str(options.rho), ...
        '  inner loop:',int2str(options.inner_loop)]);


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
        for ii=1:options.inner_loop
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
        old_alpha_list = alpha_k;
        for ii=1:Q
            % pre-calculate O(n^3)
            ske = S_k*U{ii};
            sc = S_k*c_k;
            old_alpha = alpha_k(ii);
            alpha_k(ii) = max(0,alpha_update(ske, sc, options.rho, L_k, old_alpha));
            % update new c_k
            c_k = c_k - old_alpha*U{ii} + alpha_k(ii)*U{ii};
        end
        % report phase
        if rem(i,100)==0
            % prediction (test phase)
            [pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha_k,varEst,freq,var,U);
            % [pMean, pVar] = prediction(xtest,nTest,xtrain,ytrain,nTrain,K,alpha,Q,nv,freq,var);
            MSE = mean((pMean-ytest(1:nTest)).^2);
            % plot phase
%             figName = './fig/ADMM_Temp';
%             plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName)
            % print diff
            diff_alpha = norm(old_alpha_list-alpha_k);
            obj = ytrain'*inv(c_k)*ytrain + log(det(c_k));
            disp(['Iters:',int2str(i),'  Obj:', num2str(obj),'  MSE:',num2str(MSE), ...
                '  norm2diff_alpha:',num2str(diff_alpha), ...
                '  time:',num2str(toc), ...
                '  LAMBDA matrix: ',num2str(norm(L_k,'fro')^2)]);
            % stopping signal
            if diff_alpha < 1
                alpha = alpha_k;
                return
            elseif i==options.MAX_iter
                disp('exceed max iterations.')
                alpha = alpha_k;
                return
            end
            
        end
        %%%%%%%%%%%%%%%%%%%%
        % L update
        %%%%%%%%%%%%%%%%%%%%

        % c_k is ready

        L_k = L_k + options.rho*(S_k*c_k - eyeM);

    end

    % module return final alpha
    alpha = alpha_k;
end





