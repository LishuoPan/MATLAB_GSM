function S_g = S_gradient(ytrain, S, L, C, rho, method)
%S_GRADIENT compute the gradient of S
%   Input:
%       ytrain: column vector
%       S,L,C,rho: inherit from upper function e.g. ADMM_ML.m
%       method: 0or1; 
%           0 for original(include inv(S))
%           1 for approximate(c_k replace inv(S))
    rank_one = ytrain * ytrain';
    n = length(ytrain);
    eye_M = eye(n);
    rep_one = L*C;
    rep_two = S*C*C;
    
    if method == 0
        S_g = 2*rank_one - rank_one.*eye_M ...
              - 2*inv(S) + inv(S).*eye_M ...
              + rep_one + rep_one' - rep_one.*eye_M ...
              + rho*(rep_two + rep_two' - rep_two.*eye_M) ...
              - rho*(2*C - C.*eye_M);
    else
        S_g = 2*rank_one - rank_one.*eye_M ...
              - 2*C + C.*eye_M ...
              + rep_one + rep_one' - rep_one.*eye_M ...
              + rho*(rep_two + rep_two' - rep_two.*eye_M) ...
              - rho*(2*C - C.*eye_M);
    end
    
end