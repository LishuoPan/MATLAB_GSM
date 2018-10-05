function [alpha,nv,info] = mkrm_optimize(y,Phi,U,options)
% MKRM_OPTIMIZE Multiple kernel regularization method.
%    MKRM_OPTIMIZE implements a marginal log-likelihood maximization
%    method, or equivalently, it attempts to solve non-convex problems
%    of the form
%
%      minimize  Y'*inv(Sigma(alpha,nv))*Y + log(det(Sigma(alpha,nv)))
%      subject   alpha >= 0, nv >= 0
%                nv == const,   (optional)
%
%    where Sigma(alpha,nv) = nv*I + sum_{i=1}^p alpha(i)*PHI*Pi*PHI'
%    is an affine function of hyperparameters alpha and nv which are
%    also the optimization variables. The hyperparameter nv represents
%    the noise variance, and it can be included as an optimization
%    variable or set to some positive value.
%
%    [alpha,nv,info] = MKRM_OPTIMIZE(Y,PHI,U) optimizes the marginal
%    log-likelihood function using a majorization-minimization approach.
%    Both the hyperparameters alpha and nv will be treated as variables.
%    Each iterations involves solving a convex matrix-fractional
%    minimization problem which is solved using MOSEK interior-point
%    optimizer. The input arguments consist of an N-by-n matrix PHI, an
%    N-by-1 vector Y, and a 1-by-p cell array U where U{i} is a n-by-ki
%    matrix such that Pi = U{i}*U{i}'.
%
%    [alpha,nv,info] = MKRM_OPTIMIZE(Y,PHI,U,OPTIONS) minimizes with
%    the default optimization parameters replaced by values in OPTIONS.
%    The OPTIONS struct has the following fields:
%
%       OPTIOS.ev is a logical that determines whether or not to
%       'estimate variance', i.e., the hyperparameter nv is treated as
%       a constant if OPTIONS.ev is <strong>false</strong>, and it is
%       treated as a variable if OPTIONS.ev is <strong>true</strong>.
%       The default value is <strong>true</strong>.
%
%       OPTIONS.nv is the noise variance; adds the constraint
%       nv == OPTIONS.nv if OPTIONS.ev is <strong>false</strong>, and
%       otherwise OPTIONS.nv is treated as an initial guess for the
%       hyperparameter nv. The default value is 1.0.
%
%       OPTIONS.alpha is an initial guess for the hyperparameter
%       vector alpha (a p-by-one vector). The default value is a
%       vector of ones.
%
%       OPTIONS.maxiters is the maximum number of
%       majorization-minimization iterations. The default value is
%       50 iterations.
%
%       OPTIONS.verbose specifies the verbosity level (0,1,2);
%       setting this option to 0 supresses all output to the
%       screen, 1 prints majorization-minimization progress
%       information, and 2 prints majorization-minimization
%       progress information as well as information from the
%       interior-point solver. The default value is 0.
%
%       OPTIONS.solver specifies the interior-point solver; 'mosek',
%       'sedumi', or 'sdpt3'. The default solver is MOSEK.
%
%
%
%    Copyright (C) 2014 Martin S. Andersen and Tianshi Chen
%
%    This program is free software: you can redistribute it and/or
%    modify it under the terms of the GNU General Public License as
%    published by the Free Software Foundation, either version 3 of
%    the License, or (at your option) any later version.
%
%    This program is distributed in the hope that it will be useful,
%    but WITHOUT ANY WARRANTY; without even the implied warranty of
%    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
%    General Public License for more details.
%
%    You should have received a copy of the GNU General Public License
%    along with this program.  If not, see <http://www.gnu.org/licenses/>.


%% Start timer and extract problem dimensions 

Tpre = tic;

assert(size(y,2) == 1, 'Y must be a column vector');
N = size(y,1);

assert(iscell(U),'U must be a cell array');
p = length(U);

if ~iscell(Phi)
    assert(size(Phi,1) == N, sprintf('Y must be a vector of length %i',size(Phi,1)));
    n = size(Phi,2);
else
    assert(length(Phi) == p, sprintf('PHI must me a cell array of length %i',p));
    for i = 1:p
        assert(size(Phi{i},1) == N, sprintf('PHI{%i} must have %i rows',i,N));
    end
end


%% Parse options

if isfield(options,'dimension_reduction')
    dimension_reduction = options.dimension_reduction;
    assert(islogical(dimension_reduction), 'option ''dimension_reduction'' must be a logical');
else
    dimension_reduction = true;
end
if isfield(options,'maxiters')
    maxiters = options.maxiters;
    assert(isscalar(maxiters) && maxiters >= 1,'options ''maxiters'' must a positive integer');
else
    maxiters = 50;
end
if isfield(options,'verbose')
    verbose = options.verbose;
    assert(isscalar(verbose) && verbose >= 0,'option ''verbose'' must be a nonnegative integer');
else
    verbose = 0;
end
if isfield(options,'tol')
    tol = options.tol;
    assert(isscalar(tol) && tol > 0,'option ''tol'' must be a positive scalar');
else
    tol = 1e-2;
end
if isfield(options,'nztol')
    nztol = options.nztol;
    assert(isscalar(nztol) && nztol >= 0,'option ''nztol'' must be a positive scalar');
else
    nztol = 1e-6;
end
if isfield(options,'reltol')
    reltol = options.reltol;
    assert(isscalar(reltol) && reltol >= 0,'option ''reltol'' must be a positive scalar');
else
    reltol = 1e-6;
end
if isfield(options,'ev')
    ev = options.ev;
    assert(islogical(ev),'option ''ev'' must be a logical');
else
    ev = true;
end
if isfield(options,'nv')
    nv = options.nv;
    assert(isscalar(nv) && nv >= 0,'option ''nv'' must be a nonnegative scalar');
else
    nv = 1.0;
end
if isfield(options,'c_nv')
    c_nv = options.c_nv;
    assert(isscalar(c_nv) && c_nv >= 0,'option ''c_nv'' must be a nonnegative scalar');
else
    c_nv = 0;
end
if isfield(options,'alpha')
    alpha = options.alpha;
    assert(size(alpha,1) == p, sprintf('option ''alpha'' must be a column vector of length %i',p));
    assert(size(alpha,2) == 1, 'option ''alpha'' must be a column vector');
    assert(sum(min(0,alpha)) == 0, 'option ''alpha'' must be a nonnegative vector');
else
    alpha = ones(p,1);
end
if isfield(options,'c_alpha')
    c_alpha = options.c_alpha;
    assert(size(c_alpha,1) == p, sprintf('option ''c_alpha'' must be a column vector of length %i',p));
    assert(size(c_alpha,2) == 1, 'option ''c_alpha'' must be a column vector');
    assert(sum(min(0,c_alpha)) == 0, 'option ''c_alpha'' must be a nonnegative vector');
else
    c_alpha = zeros(p,1);
end
if isfield(options,'solver')
    solver = lower(options.solver);
else
    solver = 'mosek';
end
alpha = [nv; alpha]; % hyper-parameter
if verbose, fprintf(1,'Solver: %s\n', solver), end

%% Dimension reduction via QR decomposition
%
% When the system of equations Y = Phi*x + V is overdetermined,
% i.e., there are more measurements than unknowns, it is often
% beneficial to employ a QR factorization to reduce the dimension
% of the problem.
%
% Suppose N > n+1, and consider the following QR decomposition
%
%       [Phi Y] = [Q1 Q2] [R1 R2; 0 0]
%
% where
%
%       Q1 is N-by-(n+1) matrix and Q1*Q1' = I_{n+1},
%       R1 is (n+1)-by-n matrix,
%       R2 is (n+1)-by-1 vector.
%
% Then the original marginal likelihood maximization problem
%
%       minimize   Y'*inv(Sigma)*Y + log(det(Sigma)),
%
% where Sigma = nv*I_N + sum_{i=1}^p alpha(i)*Phi*Pi*Phi', becomes
%
%       minimize   R2'*inv(Gamma)*R2 + log(det(Gamma)) + (N-n-1)*log(nv)
%
% where Gamma = nv*I_{n+1} + sum_{i=1}^p alpha(i)*R1*Pi*R1'.
%
%
Nt = N;
if ~iscell(Phi) && N > n+1 && dimension_reduction
    if verbose
        fprintf(1,'Data preprocessing: dimension reduction\n');
    end
    tmp = triu(qr([Phi y]));
    y = tmp(1:n+1,end);
    Phi = tmp(1:n+1,1:n);
    Nt = n+1;
end

%% Preprocessing the data to save the computation
% In the majorization minimization step, several matrix-matrix
% computations are repeated at each iteration. To avoid this, we
% storing intermediate matrices. The effect on runtime depends on
% the problem dimensions.
Bsqr = cell(1,p);
B = cell(1,p);
if ~iscell(Phi)
    for i = 1:p
        Bsqr{i} = Phi*U{i};
        B{i} = (Bsqr{i}*Bsqr{i}');
    end
else
    for i = 1:p
        Bsqr{i} = Phi{i}*U{i};
        B{i} = (Bsqr{i}*Bsqr{i}');
    end
end

%% Majorization Minimization

% Problem formulation in MOSEK:
%
% minimize     c'*x
% subject to   blc <= A*x <= buc
%              blx <= x <= bux
%              x in K
%
% The variable x is partitioned as
%
%    x = (t_0,alpha_0,v_0,t_1,alpha_1,v_1,...,t_p,alpha_p,v_p)
%
% where alpha_0 represents the noise variance. The objective function can
% be expressed as
%
%    c'*x = sum_i t_i + sum_i z_i*alpha_i
%
% where the vector z is the gradient of log(det(Sigma)). The constraint "x
% in K" means that
%
%    (t_0,alpha_0,v_0) in rot. quad. cone
%    (t_1,alpha_1,v_1) in rot. quad. cone
%     ..
%    (t_p,alpha_p,v_p) in rot. quad. cone
%
% When the noise variance is treated as a variable (i.e., ev = 1), the
% constraint "blc <= A*x <= buc" corresponds to the equality constraints
%
%    A*x = sum_i Phi*U{i}*v_i + v_0 = y
%
% i.e., if we let A = [A0, A1, ..., Ap], then
%
%    A0 = [0 0 I], A1 = [0 0 Phi*U{1}], ..., Ap = [0 0 Phi*U{p}].
%
% When the noise variance is treated as a constant (i.e., ev = 0), the
% constrant "blc <= A*x <= buc" corresponds to the equality constraints
%
%    A*x = [ sum_i Phi*U{i}*v_i + v_0 ] = [ y  ]
%          [ alpha_0                  ] = [ nv ].
%

Kp = zeros(1,p);
for i = 1:p, Kp(i) = size(Bsqr{i},2); end
At = cell(1,p+1);
At{1} = [sparse(Nt,2), speye(Nt)];
for i = 1:p, At{i+1} = [sparse(Nt,2) Bsqr{i}]; end
C = cell(p+1,1);
C{1} = sparse([1,2],[1,1],[2,1],Nt+2,1);
for i = 1:p, C{i+1} = sparse([1,2],[1,1],[2,1],Kp(i)+2,1); end

if strcmp(solver,'mosek')
    if exist('mosekopt')~=3, error('MOSEK is not available.'); end
    % Build MOSEK problem structure
    prob.blc = y;
    prob.buc = y;
    prob.a = cell2mat(At);
    prob.c = cell2mat(C);
    if ev == 0
        % add equality constraint if noise variance is fixed
        prob.a = [prob.a; sparse(1,2,1,1,size(prob.a,2))];
        prob.blc = [prob.blc;nv];
        prob.buc = [prob.buc;nv];
    end
    prob.blx = -inf*ones(size(prob.a,2),1); % no explicit lower bounds
    prob.bux =  inf*ones(size(prob.a,2),1); % no explicit upper bounds
    prob.cones.sub    = 1:size(prob.a,2);
    prob.cones.subptr = cumsum([1 Nt+2 Kp(1:end-1)+2]);
    [~,res] = mosekopt('symbcon echo(0)');
    prob.cones.type   = ones(p+1,1)*res.symbcon.MSK_CT_RQUAD;
elseif strcmp(solver,'sedumi')
    if exist('sedumi')~=2, error('SeDuMi is not available.'); end
    subptr = cumsum([1 Nt+2 Kp(1:end-1)+2]);
    % Build SeDuMi problem structure
    Am = cell2mat(At);
    bt = y;
    ct = cell2mat(C);
    if ev == 0
        % add equality constraint if noise variance is fixed
        Am = [Am; sparse([1,1],[1,2],[1/sqrt(2),-1/sqrt(2)],1,size(Am,2))];
        bt = [bt; nv];
    end
    K = struct('l',0,'q',[Nt+2 Kp+2],'s',0);
    pars = checkpars(struct);
    if verbose <= 1
        pars.fid = 0;  % tell SeDuMi to be quiet
    end
elseif strcmp(solver,'sdpt3')
    if exist('sqlp')~=2, error('SDPT3 is not available.'); end
    subptr = cumsum([1 Nt+2 Kp(1:end-1)+2]);
    bt = y;
    if ev == 0
        % add equality constraint if noise variance is fixed
        At{1} = [At{1}; sparse([1,1],[1,2],[1/sqrt(2),-1/sqrt(2)],1,size(At{1},2))];
        for i = 1:p
            At{i+1} = [At{i+1}; sparse(1,size(At{i+1},2))];
        end
        bt = [bt; nv];
    end
    blk{1,1} = 'q';
    blk{1,2} = [Nt+2, Kp+2];
    At = {cell2mat(At)};
    C = {cell2mat(C)};
    pars = sqlparameters;
    if verbose <= 1
        pars.printlevel = 0;  % tell SDPT3 to be quiet
    end
else
    error(sprintf('Unknown solver: %s', solver));
end

% Preallocate storage
w = zeros(p+1,1);
z = zeros(p+1,1);
objval = zeros(maxiters+1,1);
mm_iters = zeros(maxiters+1,1);
mm_cputime = zeros(maxiters+1,1);

% Print iteration info header
if verbose > 0
    fprintf(1,'%4s  %11s %5s %8s %5s\n','It.','Objective','NNZ','Rel.opt','In.it');
end

% Majorization-minimization loop
Tpre = toc(Tpre);
Tsol = tic;
for k = 1:maxiters+1
    
    % Compute Sigma and its Cholesky factorization Sigma = R'*R
    Sigma = sparse(Nt,Nt);
    for i = 2:p+1,Sigma = Sigma + alpha(i)*B{i-1};end
    Sigma = Sigma + alpha(1)*speye(Nt);
    R = chol(Sigma);  
    
    % Compute objective value
    tmp = (R'\y);
    objval(k) = norm(tmp)^2 + 2*sum(log(diag(R))) + (N-Nt)*log(max(nztol,alpha(1)));
    
    % Compute gradient of y'*inv(Sigma)*y
    tmp = (R\tmp);
    if ev == 1, w(1) = -norm(tmp)^2; end
    for i = 2:p+1
        w(i) = -norm(Bsqr{i-1}'*tmp)^2;
    end
    
    % Compute gradient of log(det(Sigma)) + (N-Nt)*log(alpha(1))
    if ev == 1, z(1) = norm(R'\eye(Nt),'fro')^2 + (N-Nt)/max(nztol,alpha(1)); end
    for i = 2:p+1
        z(i) = norm(R'\Bsqr{i-1},'fro')^2;
    end
    
    % Compute optimality measure
    g = (w+z)./max(1,(abs(w)+abs(z))/2);
    tnz = max(alpha)*nztol;
    nz = sum(alpha(2:end)>tnz);
    if ev == 1
        relopt = norm(g(alpha>tnz),'inf') + ...
                 norm(max(0,-g(alpha<=tnz)),'inf');
    else
        relopt = norm(g(2:end).*(alpha(2:end)>tnz),'inf') + ...
                 norm(max(0,-g(2:end).*(alpha(2:end)<=tnz)),'inf');
    end
    
    % Print iteration information
    if verbose > 0
        fprintf(1,'%4i  % 10.4e %5i %8.1e %5i\n',k-1,objval(k),nz,relopt,mm_iters(max(1,k-1)));
    end
    
    % Check for convergence
    if relopt < tol
        status = 0; % optimal!
        if verbose, fprintf(1,'Local minimum found.\n'), end
        break
    elseif k >= 2
        if abs(objval(k-1)-objval(k))/max(1.,abs(objval(k))) < reltol
            status = 1; % slow progress
            if verbose, fprintf(1,'Slow progress.\n'), end
            break
        end
    end
    if k == maxiters+1
        status = 2; % maxiters
        if verbose, fprintf(1,'Maximum number of iterations reached.\n'), end
        break
    end
    
    % Add optional linear cost (default is zero)
    z = z+[c_nv;c_alpha];
    
    if strcmp(solver,'mosek')
        % Update problem data
        prob.c(prob.cones.subptr+1) = z;
        % Solve with MOSEK
        if verbose <= 1
            % suppress progress info
            [~,res] = mosekopt('minimize info echo(0)',prob);
        else
            % print progress info
            [~,res] = mosekopt('minimize info',prob);
        end
        % Extract solution
        alpha = res.sol.itr.xx(prob.cones.subptr+1);
        % Extract info
        mm_cputime(k) = res.info.MSK_DINF_OPTIMIZER_TIME;
        mm_iters(k) = res.info.MSK_IINF_INTPNT_ITER;
    elseif strcmp(solver,'sedumi')
        % Update problem data
        ct(subptr) = sqrt(2)*(1+0.5*z);
        ct(subptr+1) = sqrt(2)*(1-0.5*z);
        % Solve with SeDuMi
        [X,~,sedumi_info] = sedumi(Am,bt,ct,K,pars);
        % Extract solution
        alpha = (X(subptr)-X(subptr+1))/sqrt(2);
        % Extract info
        mm_cputime(k) = sedumi_info.cpusec;
        mm_iters(k) = sedumi_info.iter;
    elseif strcmp(solver,'sdpt3')
        % Update problem data
        C{1}(subptr) = sqrt(2)*(1+0.5*z);
        C{1}(subptr+1) = sqrt(2)*(1-0.5*z);
        % Solve with SDPT3
        [~,X,~,~,sdpt3_info] = sqlp(blk,At,C,bt,pars);
        % Extract solution
        alpha = (X{1}(subptr)-X{1}(subptr+1))/sqrt(2);
        % Extract info
        mm_cputime(k) = sdpt3_info.cputime;
        mm_iters(k) = sdpt3_info.iter;
    end
end
Tsol = toc(Tsol);

%% Assign output variables and info struct
nv = alpha(1);
alpha = alpha(2:end);
info.objval = objval(1:k);
info.status = status;
info.time = [Tpre,Tsol];
info.iterations = k-1;
info.solver.str = solver;
info.solver.cputime = mm_cputime(1:k);
info.solver.iterations = mm_iters(1:k);
