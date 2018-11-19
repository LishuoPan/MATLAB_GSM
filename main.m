clc;clear;
addpath('./functions')
addpath('./data')
addpath ~/mosek/8/toolbox/r2014a

% read in data & some general setup
file_name = 'electricitydata';
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
nTest = 20;
varEst = evar(ytrain);


% generate GSM kernels
% setting part: options for generate; activate nystrom.
Nystrom_activate = 0; % 0 for deactivate nystrom, 1 for activate nystrom

%Sampling method: 0 represents fixed grids, 1 represents random.
options_gen = struct('freq_lb', 0, 'freq_ub', 0.5, ...
                 'var_lb', 0, 'var_ub', 16 / (max(xtrain) - min(xtrain)), ...
                 'Q', 300, ...
                 'nFreqCand', 300, 'nVarCand', 1, ...
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
Opt_method = 1;% 0 for DCP; 1 for ADMM; 2 for DCP&ADMM

if Opt_method == 1
    % ADMM ML Opt
        % method for gradient descent:
        % 0 for original(include inv(S))
        % 1 for approximate(c_k replace inv(S))
    options_ADMM = struct('rho', 2000, 'MAX_iter', 1000, 'nv', varEst, ...
                          'iniAlpha', 200*ones(Q,1),'gradient_method',0);

%     alpha = ADMM_ML(ytrain,K,options_ADMM);

    % this part is ADMM_ML module add plot function
    alpha = ADMM_ML_plot(xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,K,options_ADMM);
    
elseif Opt_method == 0
    % DCP Opt
    Phi = eye(nTrain);
    iniAlpha = ini_Alpha('fix', 0, Q, ytrain, K);
    options_DCP = struct('verbose',1,'ev',false, ...
                     'nv',varEst, ...
                     'dimension_reduction',true, ...
                     'c_nv',0.0, ...
                     'c_alpha', iniAlpha,...
                     'maxiters', 30);
    tic
    [alpha,nv,info] = mkrm_optimize(ytrain,Phi,L,options_DCP);
    toc
elseif Opt_method == 2
    % DCP Opt
    Phi = eye(nTrain);
    iniAlpha = ini_Alpha('fix', 0, Q, ytrain, K);
    options_DCP = struct('verbose',1,'ev',false, ...
                     'nv',varEst, ...
                     'dimension_reduction',true, ...
                     'c_nv',0.0, ...
                     'c_alpha', iniAlpha,...
                     'maxiters', 1);
    [alpha,nv,info] = mkrm_optimize(ytrain,Phi,L,options_DCP);
    [pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha,varEst,freq,var,K);
    MSE = mean((pMean-ytest(1:nTest)).^2)
    figName = './fig/DCPTemp';
    plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName);
    % ADMM ML Opt
    options_ADMM = struct('rho', 2000, 'MAX_iter', 500, 'nv', varEst, ...
                          'iniAlpha', alpha,'gradient_method',1);
    
    alpha = ADMM_ML_plot(xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,K,options_ADMM);
    
end

% prediction (test phase)
[pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha,varEst,freq,var,K);
MSE = mean((pMean-ytest(1:nTest)).^2)
% [pMean, pVar] = prediction(xtest,nTest,xtrain,ytrain,nTrain,K,alpha,Q,nv,freq,var);

% plot phase
figName = ['./fig/Temp',file_name,'Q',int2str(Q)];
plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName)





