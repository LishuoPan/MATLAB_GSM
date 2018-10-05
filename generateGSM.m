function [freq, var, Q] = generateGSM(options)
%This function generates two vectors of length Q(or nFreqCand*nVarCand)
%according to the specified arguments, including sampling strategy.

%Explanation of Q, nFreqCand and nVarCand:
%If the grids are fixed, Q is not needed, while nFreqCand and nVarCand are
%To sample randomly, only Q is needed. For fixed var, specify nVarCand as 1

if options.sampling == 'fix'
    freq_can = linspace(options.freq_lb, options.freq_ub, options.nFreqCand);
    freq_grid = repelem(freq_can, options.nVarCand);
    if options.nVarCand==1
        var_can = options.fix_var;
    else
        var_can = linspace(options.var_lb, options.var_ub, options.nVarCand);
    end
    var_grid = repmat(var_can, 1, options.nFreqCand);
elseif options.sampling == 'random'
    freq_grid = options.freq_lb + (options.freq_ub - options.freq_lb)*rand(Q,1);
    if options.nVarCand==1
        var_grid = options.fix_var;
    else
        var_grid = options.var_lb + (options.var_ub - options.var_lb)*rand(Q,1);
    end
end

freq = freq_grid;
var = var_grid;
Q = length(freq);

end