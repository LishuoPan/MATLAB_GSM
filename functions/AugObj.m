function feval = AugObj(ytrain, S, L, C, rho)
%% Compute Augmented Lagrangian
    n = length(ytrain);
    LowTri = chol(S);
    logdetS = 2*sum(log(diag(LowTri)));
    Diff = S*C - eye(n);
    feval = ytrain'*S*ytrain - logdetS ...
            + L(:)' * Diff(:) + rho/2*norm(Diff,'fro')^2;
end
