function alpha = ini_Alpha(method,fixed_value,Q,ytrain,K)
%This function initializes alpha. It can be fixed by one value(usually zero
%or one), or calculated specifically.

%If the value is fixed, ini_Alpha('fix',fixd_value,Q,[],[])
%If the value need to be computed, ini_Alpha('compute',[],Q,ytrain,K)
if method=='fix'
    iniAlpha = repmat(fixed_value,Q,1);
elseif method=='compute'
    iniAlpha = zeros(Q,1);
    sampleCovMatrix = ytrain*ytrain';
    vec1 = sampleCovMatrix(:);
    fNorm1 = norm(sampleCovMatrix, 'fro');
    for k=1:Q
        subKernel = K(k);
        vec2 = subKernel(:);
        fProduct = vec1.'*vec2;
        fNorm2 = norm(subKernel,'fro');
        iniAlpha(k) = fProduct/(fNorm1*fNorm2);
    end
    s = sum(iniAlpha);
    iniAlpha = iniAlpha/s;
end
alpha = iniAlpha;
end