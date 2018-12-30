function [step,goodness] = ArmijoStep(ytrain, S, L, C, rho, Sg_vec, d_vec)
%% Armijo Rule find the step size
    s = 1;
    n = length(ytrain);
    beta = 1/5;
    sigma = 1e-5;
    SearchMax = 100;
    d = reshape(d_vec,[n,n]);
%     Sg = S_gradient(ytrain, S, L, C, rho);
%     d = -Sg/norm(Sg,'fro');
    % strat search
    for i = 1:SearchMax
        betaM = beta^i;
        % (S+delta*Sg) matrix 
        SSea = S+betaM*s*d;
        % PD check
        [~,PD] = chol(SSea);
        if PD == 0
            fevalCur = AugObj(ytrain, S, L, C, rho);
            fevalSea = AugObj(ytrain, SSea, L, C, rho);
            RHS = -sigma*betaM*s*(Sg_vec'*d_vec);
            if fevalCur-fevalSea>=RHS
                step = betaM*s;
                goodness = 'Success Search';
                return
            end
        end
    end
    disp('Armijo Fail')
    goodness = 'Fail Search';
    step = betaM*s;
end