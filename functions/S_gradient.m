function S_g = S_gradient(ytrain, S, L, C, rho)
%S_GRADIENT compute the normalized gradient of S
%   Input:
%       ytrain: column vector
%       S,L,C,rho: inherit from upper function e.g. ADMM_ML.m
%       method: 0or1; 
%           0 for original(include inv(S))
%           1 for approximate(c_k replace inv(S))
%           2 for further approximate(S_k*c_k=I)
    n = length(ytrain);
    eye_M = eye(n);
    rank_one = ytrain * ytrain';
    indicator = norm(S*C-eye_M,'fro');
    delta = 1;
    if indicator > delta
        psi = S*C;
        zeta = -rho*eye_M + L + rho*psi;
        eta = zeta*C;
        S_g = 2*rank_one - rank_one.*eye_M ...
              - 2*pinv(S) + inv(S).*eye_M ...
              + eta + eta' - eta.*eye_M;
    elseif indicator <= delta
        psi = S*C;
        phi = (-1-rho)*eye_M + L + rho*psi;
        omega = phi*C;
        S_g = 2*rank_one - rank_one.*eye_M ...
              + omega + omega' - omega.*eye_M;
    end
end