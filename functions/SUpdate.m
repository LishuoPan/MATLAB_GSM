function [SReturn, SSubIter] = SUpdate(ytrain, S, L, C, rho, MaxIL)
    % Setting Parameters
    epsilon = 1e-15;
    n = length(ytrain);
    DecayRate = 0.95;
    % Set the gradient descent method
%     GradientMethod = 'Steep';
    GradientMethod = 'NAG';
    StepMethod = 'Armijo';
%     StepMethod = 'Diminish';
    
    % Set initial(fake) Direction & Step to start the Algorithm
    NAGGrad = 0;
    d_vec = zeros(n*n,1);
    step = 0;
    
    for ii=1:MaxIL
        SSubIter = ii;
        %%%%%%%%%%%%%%%%%%%%
        % Gradient Direction
        %%%%%%%%%%%%%%%%%%%%
        if strcmp(GradientMethod, 'Steep')
            % Compute the Steepest gradient normalized S gradient & update S
            Sg = S_gradient(ytrain, S, L, C, rho);
            % Reform the gradient into a vector, and normalize it. d_vec is
            % descent direction (nagetive of gradient)
            Sg_vec = Sg(:);
            d_vec = -(Sg_vec/norm(Sg_vec));
        elseif strcmp(GradientMethod, 'NAG')
            % Save the Sg_vec for Armijo Use
            Sg = S_gradient(ytrain, S, L, C, rho);
            Sg_vec = Sg(:);
            % Reform the d_vec into matrix
            d = reshape(d_vec,[n,n]);
            % Compute the NAG gradient. Note that the d is descent
            % direction. g = -d.
            SNAGNext = S + step*DecayRate*d;
            NAGGrad = DecayRate*NAGGrad + S_gradient(ytrain, SNAGNext, L, C, rho);
            % Reform the gradient into a vector, and normalize it. d_vec is
            % descent direction (nagetive of gradient)
            NAGGrad_vec = NAGGrad(:);
            d_vec = -(NAGGrad_vec/norm(NAGGrad_vec));
        end  
        %%%%%%%%%%%%%%%%%%%%
        % Step Size
        %%%%%%%%%%%%%%%%%%%%
        if strcmp(StepMethod, 'Armijo')
            %Armijo Rule to decide the step size
            [step,goodness] = ...
                ArmijoStep(ytrain, S, L, C, rho, Sg_vec, d_vec);
        elseif strcmp(StepMethod, 'Diminish')
            % Diminishing step size. This is not valid for ADMM algorithm
            c = 1;
            step = (1e-10)/ii*c;
        end
        % update S
        d = reshape(d_vec,[n,n]);
        Z = S + step * d;
        % Inner loop stopping criteria
        if norm(Z-S,'fro')<epsilon
            S = Z;
            break
        end
        % Update S
        S = Z;
    end
    SReturn = S;
end