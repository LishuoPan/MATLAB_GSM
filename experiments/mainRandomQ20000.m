clc;clear;
addpath('./data')
addpath('./functions')

%Mosek setup
javaaddpath('C:\Program Files\Mosek\8\tools\platform\win64x86\bin/mosekmatlab.jar')
addpath('C:\Program Files\Mosek\8\toolbox\r2014a')
%addpath ~/mosek/8/toolbox/r2014a

% Read in data & some general setup
file_name = 'electricitydata';
disp(['Simulation on ',file_name]);
[xtrain, ytrain, xtest, ytest] = load_data(file_name);
nTrain = length(xtrain);
nTest = 20;
varEst = evar(ytrain);

% Generate GSM kernels
% Setting part: options for generate; activate nystrom.
Nystrom_activate = 0; % 0 for deactivate nystrom, 1 for activate nystrom

% Sampling method: 0 represents fixed grids, 1 represents random.
options_gen = struct('freq_lb', 0, 'freq_ub', 0.5, ...
                 'var_lb', 0, 'var_ub', 0.15, ...
                 'Q', 20000, ...
                 'nFreqCand', 20000, 'nVarCand', 20000, ...
                 'fix_var', 0.001,...
                 'sampling', 1 );

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


%Hyper-parameters Optimization
tic;
% DCP

Phi = eye(nTrain);
%Initialize alpha. First argument 0:fix, 1: compute, 2: random.
iniAlpha = ini_Alpha(0, 0, Q, ytrain, K);
options_DCP = struct('verbose',1,'ev',false, ...
                 'nv',varEst, ...
                 'dimension_reduction',true, ...
                 'c_nv',0.0, ...
                 'c_alpha', iniAlpha,...
                 'maxiters', 30);
[alpha,nv,info] = mkrm_optimize(ytrain,Phi,L,options_DCP);

time_record = toc;

% Prediction
[pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha,varEst,freq,var,K);

% Plot and save figure
sampling = cell(1,2); sampling{1}='fixed'; sampling{2}='random';sample_method = sampling{options_gen.sampling+1};
figName = ['./fig/',file_name,'_Q',int2str(Q),'_',sample_method,'.fig'];
plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName);

% Record MSE
MSE = mean((pMean-ytest(1:nTest)).^2);

% Save info
save(['./fig/',file_name,'_',sample_method,'_Q',int2str(Q),'.mat'], 'MSE', 'info','time_record');