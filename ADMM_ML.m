function alpha = ADMM_ML(ytrain,U,options)

Q = numel(U);
d = length(ytrain);
eyeM = eye(d);
rank_one_M = ytrain*ytrain';

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
    % iter display
    disp(['Iteration ',int2str(i),' is running...']);

    %%%%%%%%%%%%%%%%%%%%
    % S update
    %%%%%%%%%%%%%%%%%%%%
    step = 1e-16;
    for ii=1:1000
        gradient = rank_one_M - inv(S_k) + L_k*c_k + options.rho*S_k*c_k*c_k-options.rho*c_k;
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

        first_term = - trace( (K_tilde*U{ii}+options.nv*U{ii}) * (S_k'*S_k) - temp ) / trace(temp'*temp);
        second_term = - trace(L_k'*temp) / (options.rho * trace(temp'*temp));
        alpha_k(ii) = max(0, first_term+second_term);

        if ii<Q
            K_tilde = K_tilde + alpha_k(ii)*U{ii}  - alpha_k(ii+1)*U{ii+1};% update K_tilde for the next iteration
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%
    % L update
    %%%%%%%%%%%%%%%%%%%%

    % c_k update first
    c_k = C_matrix(alpha_k, U, options.nv, eyeM);  
    L_k = L_k + options.rho*(S_k*c_k - eyeM);
    % display the fnorm of BIG LAMBDA as a reference of convergence
    disp(['BIG LAMBDA: ',int2str(norm(L_k,'fro')^2)])

    % give back the K_tilde to the next iteration(NO FIRST WEIGHT.)
    K_tilde = K_tilde - alpha_k(1)*U{1} + alpha_k(Q)*U{Q};
    %
     
end

alpha = alpha_k;
end





