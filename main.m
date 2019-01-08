clc;clear;
addpath('./functions')
addpath('./functions/l1_ls_matlab')
addpath('./data')
addpath ~/mosek/8/toolbox/r2014a

% read in data & some general setup
file_name = 'electricitydata';
disp(['Simulation on ',file_name]);
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
        % 2 for further approximate(S_k*c_k=I)
    [iniAlpha_Pdg, goodness] = alphaIniFromPeriodogram(ytrain, Q, freq, var(1));

    % ADMM ML Opt
    options_ADMM = struct('rho', 100, 'rho_dual', 50, 'MaxIL', 500, 'mu', 1e-6, 'MAX_iter', 300000, 'nv', varEst, ...
                          'iniAlpha', iniAlpha_Pdg);
%     options_ADMM = struct('rho', 100, 'rho_dual', 1, 'MaxIL', 300, 'mu', 1e-7, 'MAX_iter', 5000, 'nv', varEst, ...
%                           'iniAlpha', 200*ones(Q,1));
    
    [alpha, AugObjEval, OriObjEval] = ADMM_ML_plot(xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,K,options_ADMM);

    
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
    [alpha_DCP,nv,info] = mkrm_optimize(ytrain,Phi,L,options_DCP);
    [pMean_DCP, pVar_DCP] = prediction(xtrain,xtest,ytrain,nTest,alpha_DCP,varEst,freq,var,K);
    MSE_DCP = mean((pMean_DCP-ytest(1:nTest)).^2);
    c_k = C_matrix(alpha_DCP,K,varEst,eye(length(ytrain)));
    L = chol(c_k);
    inv_LT_y = pinv(L')*ytrain;
    obj_DCP = inv_LT_y'*inv_LT_y + log(det(L')) + log(det(L));
    disp(' ');
    disp(['MSE of DCP:',num2str(MSE_DCP), '  Obj_DCP:', sprintf('%0.5e',obj_DCP)]);
    disp(' ');
%     figName = './fig/DCPTemp';
%     plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName);

    % ADMM ML Opt
    options_ADMM = struct('rho', 30, 'rho_dual', 10, 'MaxIL', 500, 'mu', 1e-7, 'MAX_iter', 50000, 'nv', varEst, ...
                          'iniAlpha', alpha_DCP);
    
    alpha = ADMM_ML_plot(xtrain,xtest,ytrain,ytest,nTest,varEst,freq,var,K,options_ADMM);
    
end

if Opt_method == 2
    % prediction (test phase)
    [pMean_final, pVar_final] = prediction(xtrain,xtest,ytrain,nTest,alpha,varEst,freq,var,K);
    MSE_final = mean((pMean_final-ytest(1:nTest)).^2);
    % [pMean, pVar] = prediction(xtest,nTest,xtrain,ytrain,nTrain,K,alpha,Q,nv,freq,var);

    % plot phase
    figName = ['./fig/Temp',file_name,'Q',int2str(Q)];
    plot_save_compare(xtrain,ytrain,xtest,ytest,nTest,pMean_DCP,pVar_DCP,pMean_final,figName,file_name);

else
    [pMean, pVar] = prediction(xtrain,xtest,ytrain,nTest,alpha,varEst,freq,var,K);
    MSE = mean((pMean-ytest(1:nTest)).^2);
    figName = ['./fig/Temp',file_name,'Q',int2str(Q)];
    plot_save(xtrain,ytrain,xtest,ytest,nTest,pMean,pVar,figName,file_name);
end



