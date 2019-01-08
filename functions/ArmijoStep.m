function [step,goodness] = ArmijoStep(ytrain, S, L, C, rho, Sg_vec, d_vec)
%% Armijo Rule find the step size
    s = 1e-4;
    n = length(ytrain);
    beta = 1/5;
    sigma = 1e-5;
    SearchMax = 100;
    d = reshape(d_vec,[n,n]);
    % strat search
    for i = 0:SearchMax
        betaM = beta^i;
        % (S+delta*Sg) matrix 
        SSea = S+betaM*s*d;
        % PD check
        [~,PD] = chol(SSea);
        if PD == 0
            % function evalue at S
            fevalCur = AugObj(ytrain, S, L, C, rho);
            % function evalue at (S+delta*Sg)
            fevalSea = AugObj(ytrain, SSea, L, C, rho);
            RHS = -sigma*betaM*s*(Sg_vec'*d_vec);
            % Stopping criteria
            if fevalCur-fevalSea>=RHS
                step = betaM*s;
                goodness = 'Success Search';
                return
            end
        end
    end
    % if Armijo Fail, print Fail
    disp('Armijo Fail')
    goodness = 'Fail Search';
    step = betaM*s;
end