clc;clear;
addpath('../functions')
addpath('../functions/l1_ls_matlab')
addpath('../data')
addpath ~/mosek/8/toolbox/r2014a

% Read in data & some general setup
file_name = 'unemployment';
disp(['Simulation on ',file_name]);
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
nTest = 20;
varEst = evar(ytrain);


% Generate GSM kernels
% Setting part: options for generate; activate nystrom.
Nystrom_activate = 0; % 0 for deactivate nystrom, 1 for activate nystrom

%Sampling method: 0 represents fixed grids, 1 represents random.
options_gen = struct('freq_lb', 0, 'freq_ub', 0.5, ...
                 'var_lb', 0, 'var_ub', 16 / (max(xtrain) - min(xtrain)), ...
                 'Q', 500, ...
                 'nFreqCand', 500, 'nVarCand', 1, ...
                 'fix_var', 0.001, 'sampling', 0 );

[freq, var, Q] = generateGSM(options_gen); % the length of freq or var is Q we need

if Nystrom_activate == 0
    K = kernelComponent(freq, var, xtrain, xtrain);
    L = cell(1,Q);
    for kk =1:Q
        L{kk} = (cholcov(K{kk})).';
    end
else
    % Kernel matrix low rank approximation
    nys_sample = ceil(length(xtrain)/20);
    [L,K] = Nystrom(xtrain,freq,var,nys_sample);
end

% Hyperpara Opt

% 1-step DCP Opt
Phi = eye(nTrain);
iniAlpha = ini_Alpha('fix', 0, Q, ytrain, K);
% DCP settings
options_DCP = struct('verbose',1,'ev',false, ...
                 'nv',varEst, ...
                 'dimension_reduction',true, ...
                 'c_nv',0.0, ...
                 'c_alpha', iniAlpha,...
                 'maxiters', 1);
% DCP step
[alpha_DCP,nv,info] = mkrm_optimize(ytrain,Phi,L,options_DCP);
[pMean_DCP, pVar_DCP] = prediction(xtrain,xtest,ytrain,nTest,alpha_DCP,varEst,freq,var,K);


% ADMM settings
% MaxIL: numbers of Internal iteraions in gradient method; 
% MAX_iter: numbers of total outer iterations.
options_ADMM = struct('rho', 100, 'rho_dual', 50, 'MaxIL', 1000, 'mu', 1e-6, 'MAX_iter', 30000, 'nv', varEst, ...
                      'iniAlpha', alpha_DCP);
% ADMM step
[alpha, AugObjEval, OriObjEval, Gap] = ADMM_ML(xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,K,options_ADMM);
% Plot the convergence criteria
figure;plot(AugObjEval);title('Iterations v.s. Augmanted Objective');xlabel('iterations');ylabel('Aug Obj');
figure;plot(OriObjEval);title('Iterations v.s. Original Objective');xlabel('iterations');ylabel('Original Obj');
figure;plot(Gap);title('Iterations v.s. Gap');xlabel('iterations');ylabel('Gap');
figure;bar(alpha);title('alpha after ADMM');xlabel('index');ylabel('alpha value');


% Prediction
[pMean_final, pVar_final] = prediction(xtrain,xtest,ytrain,nTest,alpha,varEst,freq,var,K);
MSE_final = mean((pMean_final-ytest(1:nTest)).^2);


% Plot phase
figName = ['./fig/Temp',file_name,'Q',int2str(Q)];
plot_save_compare(xtrain,ytrain,xtest,ytest,nTest,pMean_DCP,pVar_DCP,pMean_final,figName,file_name);



