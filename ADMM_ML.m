function alpha = ADMM_ML(ytrain,U,options)

Q = numel(U);
d = length(ytrain);
eyeM = eye(d);
rank_one_M = ytrain*ytrain';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% the initial input
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%inital value
S_k = eye(d);
% alpha_k = ones(Q,1);
Good_Start = 0;
if Good_Start == 1
    load('5iterDCP20000alpha200X100.mat','alpha');
    alpha_k = alpha;
else
    alpha_k = 200*ones(Q,1);
end
% add noise to alpha
%alpha_k = alpha_k + 0.03*rand(Q,1);

L_k = eye(d);

% the first K_tilde put into the process(NO FIRST WEIGHT.)
K_tilde = 0 ;
for i = 2:Q
    K_tilde = K_tilde + alpha_k(i)*U{i};
end

% the first c_k put into the process
c_k = 0;
for i = 1:Q
        c_k = c_k + alpha_k(i)*U{i};
end
c_k = c_k + options.nv*eyeM;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% START ADMM ITERATION UPDATE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['The hyperparameter rho = ',sprintf('%0.20e',options.rho)]);
% disp(['The Lambda = ',sprintf('%0.10f',LAMBDA)]);
% eval(['num_survived_alpha',int2str(srch),'=' ,'[];']);
% survivor = [];

e1 = zeros(d,1);
e1(1) = 1;
%alpha_k = ones(Q,1);
for i= 1:options.MAX_iter
    disp(['Iteration ',int2str(i),' is running...']);
    % a record of survivor in each interation
    
    
    %
    % S update
    % inverse update
    % S_k = invToeplitz(c_k(1,:));
    
    if i<=5
        S_k = inv(rank_one_M+options.rho*c_k)*( (1+options.rho)*eyeM - L_k);
        [~,PD] = chol(S_k);
        disp(['if the S is PD(0 is true): ',int2str(PD)]);
    end
    
    if i>5
        step = 1e-16;
        for ii=1:1000
            gradient = rank_one_M - inv(S_k) + L_k*c_k + options.rho*S_k*c_k*c_k-options.rho*c_k;
            S_k = S_k - step * gradient;
        end
        [~,PD] = chol(S_k);
        disp(['if the S is PD(0 is true): ',int2str(PD)]);
    end
    %
    % alpha update
    
    for ii=1:Q
        temp = S_k*U{ii};
%         alpha_k(ii) = max(0,-( (rho*trace((K_tilde+nv*eyeM)'*S_k'*temp-temp) ...
%             +trace(L_k'*temp)) / (rho*trace(temp'*temp)+2*LAMBDA) ) );
        first_term = - trace( (K_tilde*U{ii}+options.nv*U{ii}) * (S_k'*S_k) - temp ) / trace(temp'*temp);
        second_term = - trace(L_k'*temp) / (options.rho * trace(temp'*temp));
        alpha_k(ii) = max(0, first_term+second_term);
%         disp(-( (rho*trace(K_tilde+nv*eyeM*S_k*temp-temp) ...
%             +trace(L_k'*temp)) / (rho*trace(temp'*temp)) ));
        if ii<Q
            K_tilde = K_tilde + alpha_k(ii)*U{ii}  - alpha_k(ii+1)*U{ii+1};% update K_tilde for the next iteration
        end
    end
    %record the number of survivor;
%     survivor = [survivor, survivor_tracker(alpha_k)];
    
%     figure;
%     if rem(i,50)==0
%         read_alpha_and_plot;
%     end
    %
    
    
    % L update
    % c_k update
    c_k = 0;
    for ii = 1:Q % this c_k is also for next iteration.
        c_k = c_k + alpha_k(ii)*U{ii};
    end
    c_k = c_k + options.nv*eyeM;
    
    L_k = L_k + options.rho*(S_k*c_k - eyeM);
    
    disp(['BIG LAMBDA: ',int2str(norm(L_k,'fro')^2)])
%     [~,PD] = chol(L_k);
%     disp(['if the L_k is PD(0 is true): ',int2str(PD)]);
    % give back the K_tilde to the next iteration(NO FIRST WEIGHT.)
    K_tilde = K_tilde - alpha_k(1)*U{1} + alpha_k(Q)*U{Q};
    %
    
    
end

alpha = alpha_k;
end






%{    
    % First S update method
    %{
    A = rho*c_k*c_k;
    B = rank_one_M + L_k*c_k-rho*c_k;
    C = eyeM;
    
    fun = @(S)S*A*S+B*S-C;
    options = optimoptions('fsolve','Display','final-detailed','MaxFunEvals',1.e+10,'MaxIterations',10000);
    S0 = S_k; % warm start
    S_k = fsolve(fun,S0,options);
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    %}
    
    % Second S update method
    %{
    B = rank_one_M + L_k*c_k-rho*c_k;
    S_k = (1-rho)*pinv(B);
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    %}
    
    
    % Third S update method
    %{
    B = rank_one_M + L_k*c_k;
    S_k = pinv(B);
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    %}
    
    % Forth S update method
    %{
    S_k = (1+rho)*pinv(rank_one_M+L_k*c_k+rho*c_k);
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    %}
    
    % Fifth S update method
    %{
    S_k = inv(rank_one_M+rho*c_k)*( (1+rho)*eyeM - L_k);
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    %}
    % inverse update
    S_k = invToeplitz(c_k(1,:));
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    
    % fast S update
    CKCK = c_k*c_k;
    fun = @(s)(rho*convertToeplitz(s')*CKCK+rank_one_M+L_k*c_k-rho*c_k)*s-e1;
    options = optimoptions('fsolve','Display','final-detailed');
    s0 = S_k(:,1);
    s_k = fsolve(fun,s0,options);
    S_k = convertToeplitz(s_k');
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    %

    % CVX for solving S
    cvx_begin
    variable S_k(d,d) semidefinite
    minimize( (ytrain' * S_k * ytrain - log_det(S_k)+...
        sum(sum( L_k.* (S_k*c_k-eyeM) ))+...
        rho/2 * square_pos(norm(S_k*c_k-eyeM,'fro')) ) ) % this can be better expressed.
    cvx_end
    [~,PD] = chol(S_k);
    disp(['if the S is PD(0 is true): ',int2str(PD)]);
    %
%%%%%%%%%%%%%%%%%  
%    % A FAKE S UPDATE(This part is just for debuging)
%     S_k = inv(rank_one_M+rho*c_k)*( (1+rho)*eyeM + L_k*L_k');
%     [~,PD] = chol(S_k);
%     disp(['if the S is PD(0 is true): ',int2str(PD)]);
%     
%%%%%%%%%%%%%%%%    
%}  

