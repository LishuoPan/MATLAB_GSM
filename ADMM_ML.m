function alpha = ADMM_ML(ytrain,U,options)

Q = numel(U);
d = length(ytrain);
eyeM = eye(d);
rank_one_M = ytrain*ytrain';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the initial input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

alpha_k = options.iniAlpha;

L_k = eye(d);

c_k = C_matrix(alpha_k, U, options.nv, eyeM);
S_k = inv(c_k);

% the first K_tilde put into the process(NO FIRST WEIGHT.)
K_tilde = 0 ;
for i = 2:Q
    K_tilde = K_tilde + alpha_k(i)*U{i};
end


% the first c_k put into the process


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% START ADMM ITERATION UPDATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['The hyperparameter rho = ',sprintf('%0.20e',options.rho)]);


e1 = zeros(d,1);
e1(1) = 1;
%alpha_k = ones(Q,1);
for i= 1:options.MAX_iter
    disp(['Iteration ',int2str(i),' is running...']);
    % a record of survivor in each interation
    
    
    %
    % S update
    step = 1e-16;
    for ii=1:1000
        gradient = rank_one_M - inv(S_k) + L_k*c_k + options.rho*S_k*c_k*c_k-options.rho*c_k;
        S_k = S_k - step * gradient;
    end
    [~,PD] = chol(S_k);
    if PD ~= 0
        disp(['if the S is PD(0 is true): ',int2str(PD)]);
    end
    %
    % alpha update    
    for ii=1:Q
        temp = S_k*U{ii};

        first_term = - trace( (K_tilde*U{ii}+options.nv*U{ii}) * (S_k'*S_k) - temp ) / trace(temp'*temp);
        second_term = - trace(L_k'*temp) / (options.rho * trace(temp'*temp));
        alpha_k(ii) = max(0, first_term+second_term);

        if ii<Q
            K_tilde = K_tilde + alpha_k(ii)*U{ii}  - alpha_k(ii+1)*U{ii+1};% update K_tilde for the next iteration
        end
    end

    % L update
    % c_k update
    c_k = C_matrix(alpha_k, U, options.nv, eyeM);  
    L_k = L_k + options.rho*(S_k*c_k - eyeM);
    
    disp(['BIG LAMBDA: ',int2str(norm(L_k,'fro')^2)])

    % give back the K_tilde to the next iteration(NO FIRST WEIGHT.)
    K_tilde = K_tilde - alpha_k(1)*U{1} + alpha_k(Q)*U{Q};
    %
     
end

alpha = alpha_k;
end





