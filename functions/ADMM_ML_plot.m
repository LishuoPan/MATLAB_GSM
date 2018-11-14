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

    % the first K_tilde put into the process(NO FIRST WEIGHT and noise term.)
    K_tilde = c_k - alpha_k(1)*U{1} - options.nv*eyeM;

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
            temp = S_k*U{ii};
            % divide the update close form into 2 terms
            first_term = - trace( (K_tilde*U{ii}+options.nv*U{ii}) * (S_k'*S_k) - temp ) / trace(temp'*temp);
            second_term = - trace(L_k'*temp) / (options.rho * trace(temp'*temp));
            alpha_k(ii) = max(0, first_term+second_term);
            % after update each alpha, update K_tilde.
            % K_tilde in: 0XXX...X; out: XXX...X0; 0 means kernel is omitted.
            if ii<Q
                K_tilde = K_tilde + alpha_k(ii)*U{ii}  - alpha_k(ii+1)*U{ii+1};% update K_tilde for the next iteration
            end
        end
        if rem(i,50)==0
            % iter display
            disp(['Iteration ',int2str(i),' is running...']);
            % prediction (test phase)
            [pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha_k,varEst,freq,var,U);
            % [pMean, pVar] = prediction(xtest,nTest,xtrain,ytrain,nTrain,K,alpha,Q,nv,freq,var);
            MSE = mean((pMean-ytest(1:nTest)).^2)
            % plot phase
            figName = './fig/ADMM_Temp';
            plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName)
        end
        %%%%%%%%%%%%%%%%%%%%
        % L update
        %%%%%%%%%%%%%%%%%%%%

        % c_k update first
        c_k = C_matrix(alpha_k, U, options.nv, eyeM);  
        L_k = L_k + options.rho*(S_k*c_k - eyeM);
        if rem(i,50)==0
            % display the fnorm of BIG LAMBDA as a reference of convergence
            disp(['BIG LAMBDA: ',int2str(norm(L_k,'fro')^2)])
        end
        % give back the K_tilde to the next iteration(NO FIRST WEIGHT.)
        K_tilde = K_tilde - alpha_k(1)*U{1} + alpha_k(Q)*U{Q};

    end

    % module return final alpha
    alpha = alpha_k;
end





