function [step,goodness] = ArmijoStep(ytrain, S, L, C, rho)
%% Armijo Rule find the step size
    s = 1;
    beta = 1/2;
    sigma = 1e-3;
    SearchMax = 1000;
    Sg = S_gradient(ytrain, S, L, C, rho);
    d = -Sg/norm(Sg,'fro');
    % strat search
    for i = 1:SearchMax
        betaM = beta^i;
        SSea = S+betaM*s*d;
        [~,PD] = chol(SSea);
        if PD == 0
            fevalCur = AugObj(ytrain, S, L, C, rho);
            fevalSea = AugObj(ytrain, SSea, L, C, rho);
            RHS = -sigma*betaM*s*(-norm(Sg,'fro'));
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