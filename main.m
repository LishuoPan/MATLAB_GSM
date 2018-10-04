clc;clear;
addpath('./data')
addpath ~/mosek/8/toolbox/r2014a

% read in data & some general setup
file_name = 'electricitydata';
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
varEst = evar(ytrain);


% generate GSM kernels
options_gen = struct('freq_lb', 0, 'freq_ub', 0.5, ...
                 'var_lb', 0, 'var_ub', 16 / (max(xtrain) - min(xtrain)), ...
                 'Q', 200, ...
                 'nFreqCand', 200, 'nVarCand', 1, ...
                 'fix_var', 0.01, 'sampling', 'fix' );

[freq, var, Q] = generateGSM(options_gen); % the length of freq or var is Q we need
K = kernelComponent(freq, var, xtrain);

% Kernel matrix low rank approximation


% Hyperpara Opt

% ADMM ML
options_ADMM = struct('rho', 2000, 'MAX_iter', 10000);

% DCP
Phi = eye(nTrain);
iniAlpha = ini_Alpha('fix', 1, Q, ytrain, K);
options_DCP = struct('verbose',1,'ev',false, ...
                 'nv',varEst, ...
                 'dimension_reduction',true, ...
                 'c_nv',0.0, ...
                 'c_alpha', iniAlpha,...
                 'maxiters', 30);
[alpha,nv,info] = mkrm_optimize(ytrain,Phi,K,options_DCP);


% prediction (test phase)






