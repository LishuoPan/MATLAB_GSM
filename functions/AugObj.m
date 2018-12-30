function feval = AugObj(ytrain, S, L, C, rho)
%% Compute Augmented Lagrangian
    n = length(ytrain);
    LowTri = chol(S)';
    LT_y = LowTri'*ytrain;
    logdetS = 2*sum(log(diag(LowTri)));
    Diff = S*C - eye(n);
    feval = LT_y'*LT_y - logdetS ...
            + L(:)' * Diff(:) + rho/2*norm(Diff,'fro')^2;
end
