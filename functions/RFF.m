function[U_RFF,K_RFF] = RFF(xtrain, freq_grid,var_grid, R)
%This function will calculate the (2R*n) random Fourier feature matrix

U_RFF = cell(1,length(freq_grid));
K_RFF = cell(1,length(freq_grid));

%sample f
for ii= 1:length(freq_grid)
    
    u = rand(R,1)>0.5;
    f1 = freq_grid(ii) + var_grid(ii)*randn(R,1);
    f2 = -freq_grid(ii) + var_grid(ii)*randn(R,1);
    f = (u .* f1 + (1-u) .* f2);
    %Make sure that frequency 0 is sampled
    f(1) = 0;

    C = cos(2*pi*xtrain*(f'))/sqrt(R);
    S = sin(2*pi*xtrain*(f'))/sqrt(R);
    
    PHI = [C,S];
    U_RFF{ii} = PHI;
    K_RFF{ii} = PHI*PHI';
    
end


