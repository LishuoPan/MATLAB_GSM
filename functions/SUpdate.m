function SReturn = SUpdate(ytrain, S, L, C, rho, MaxIL)
    epsilon = 1e-08;
    n = length(ytrain);
    for ii=1:MaxIL
        % compute normalized S gradient & update S
        Sg = S_gradient(ytrain, S, L, C, rho);
        Sg_vec = Sg(:);
        d_vec = -(Sg_vec/norm(Sg_vec));
        %Armijo Rule to decide the step size
        [step,goodness] = ...
            ArmijoStep(ytrain, S, L, C, rho, Sg_vec, d_vec);
        % update S
        d = reshape(d_vec,[n,n]);
        Z = S + step * d;
        % Inner loop stopping criteria
        if norm(Z-S,'fro')<epsilon
            S = Z;
            break
        end
        % Update S
        S = Z;
    end
    SReturn = S;
end