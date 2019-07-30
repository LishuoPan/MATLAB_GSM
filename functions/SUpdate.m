function [SReturn, SSubIter] = SUpdate(ytrain, S, L, C, rho, MaxIL)
    % Setting Parameters
    epsilon = 1e-20;
    n = length(ytrain);
    DecayRate = 0.98;
    % Set the gradient descent method
%     GradientMethod = 'Steep';
    GradientMethod = 'NAG';
    StepMethod = 'Armijo';
%     StepMethod = 'Diminish';
    
    % Set initial(fake) Direction & Step to start the Algorithm
    NAGGrad = 0;
    d_vec = zeros(n*n,1);
    step = 0;
    
    % Lookahead Setting
    k = 5;
    gamma = 0.5;
    
    for ii=1:MaxIL
        SSubIter = ii;
        % Synchronize Theta with S.
        Theta = S;
        % Start Lookahead Search
        for iii = 1:k
            %%%%%%%%%%%%%%%%%%%%
            % Gradient Direction
            %%%%%%%%%%%%%%%%%%%%
            if strcmp(GradientMethod, 'Steep')
                % Compute the Steepest gradient normalized S gradient & update S
                Thetag = S_gradient(ytrain, Theta, L, C, rho);
                % Reform the gradient into a vector, and normalize it. d_vec is
                % descent direction (nagetive of gradient)
                Thetag_vec = Thetag(:);
                d_vec = -(Thetag_vec/norm(Thetag_vec));
            elseif strcmp(GradientMethod, 'NAG')
                % Save the Sg_vec for Armijo Use
                Thetag = S_gradient(ytrain, Theta, L, C, rho);
                Thetag_vec = Thetag(:);
                % Reform the d_vec into matrix
                d = reshape(d_vec,[n,n]);
                % Compute the NAG gradient. Note that the d is descent
                % direction. g = -d.
                ThetaNAGNext = Theta + step*DecayRate*d;
                NAGGrad = DecayRate*NAGGrad + S_gradient(ytrain, ThetaNAGNext, L, C, rho);
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
                    ArmijoStep(ytrain, Theta, L, C, rho, Thetag_vec, d_vec);
            elseif strcmp(StepMethod, 'Diminish')
                % Diminishing step size. This is not valid for ADMM algorithm
                c = 1;
                step = (1e-10)/iii*c;
            end

            % update Theta
            d = reshape(d_vec,[n,n]);
            Theta = Theta + step * d;
        
        end
        
        % interpolation to generate new S into Z
        Z = S + gamma*(Theta - S);
        
        % Inner loop early stopping criteria
        if norm(Z-S,'fro')<epsilon
            S = Z;
            break
        end
        
        % let Z be the new S
        S = Z;
        
    end
    SReturn = S;
end