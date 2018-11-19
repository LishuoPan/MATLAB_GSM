function alpha_k_next = alpha_update(ske, sc, rho, L_k, alpha_k)
%ALPHA UPDATA 
%complexity O(n^2) module.

    num = trace(ske) - ske(:)'*sc(:) - 1/rho*(L_k(:)'*ske(:));
    den = ske(:)'*ske(:);
    alpha_k_next = alpha_k+num/den;
end