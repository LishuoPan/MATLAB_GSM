function obj = ML_obj(C, y)
    L = chol(C);
    inv_LT_y = pinv(L')*y;
    obj = inv_LT_y'*inv_LT_y + log(det(L')) + log(det(L));
end
