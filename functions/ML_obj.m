function obj = ML_obj(C, y)
    % L is the lower triangle matrix after chol decomp
    L = chol(C)';
    inv_L_y = pinv(L)*y;
    obj = inv_L_y'*inv_L_y + 2*sum(log(diag(L)));
end
