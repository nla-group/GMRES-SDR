function [x, resvec, out] = gcrodr(A, b, opt)
% [X, resvec, out] = gcrodr(A, b, opt)
%
% We are solving systems of the form Ax=b using the recycle GMRES method described in
% [Michael L. Parks, Eric de Sturler, Greg Mackey, Duane Johnson, and Spandan Maiti, 
%  Recycling Krylov Subspaces for Sequences of Linear Systems, 
%  SIAM Journal on Scientific Computing, 28(5), pp. 1651-1674, 2006.]
% Right and left preconditioning are acceptable
%
% DATA DICTIONARY
% INPUTS:
%   A   Coefficient matrix or a matvec procedure
%   b   The right-hand side
%   opt Structure containing optional inputs or options
%   opt.M1  Coefficient matrix representing left preconditioner which must 
%           be inverted using backslash or a procedure which applies the
%           preconditioner inverse. If a matrix is passed in, then 
%           backslash is used.  If opt.M is a procedure, it is assumed that
%           the procedure is for solving a linear system involving M.
%   opt.M2  The same but a right preconditioner
%   opt.tol Optional convergence tolerance.  Default: 1e-8
%   opt.nrestarts  Optional maximum restart cycles. Default: 10
%   opt.cyclelength  Optional maximum iterations per cycle.  Default: 50
%   opt.recyclealg  Different choices for how to downselect at the end of a
%                   cycle: harmvecs (default), ritzvecs, OT (optimal truncation)
%   opt.X0   Optional initial approxations.  Size(opt.X0) =
%            [n,length(shifts)]
%   opt.U    Optional initial recycle space.  Default: [].  Since this
%            function is called recursively, U will be declared as a global
%            variable.  If there is already a global U defined, the space
%            stored in that variable will be used.  If not, the default []
%            will be used.  If the global U exists AND opt.U exists, then
%            we will overwrite U = opt.U;
%   opt.isOutU   Do we output U as out.U?
%   opt.k    Specifies the maximum dimension of the recycled subspace.
%            Default: 10
%   opt.isOutNMV   Do we output the total number of matvecs nmv?
%   opt.isComplex   Is the matrix complex?
%   opt.isPrintStatus   Display iteration status indicator 
% OUTPUTS:
%   X       Columns are the approximate solutions for all shifted systems
%   resvec  Cell array containing the resvecs
%   out     Structure containing any optional outputs.
%
% Kirk Soodhalter
% 09-Jul-2014 23:52:02
% kirk@math.soodhalter.com

    n = length(b);

    if ~exist('opt','var')
        opt = struct(); %empty structure
    end

    if isfield(opt,'tol')
        tol = opt.tol;
    else
        tol = 1e-8;
    end

    if isfield(opt,'nrestarts')
        nrestarts = opt.nrestarts;
    else
        nrestarts = 10;
    end

    if isfield(opt,'x0')
        x0 = opt.x0;
    else
        x0 = zeros(n,1);
    end

    if isfield(opt,'cyclelength')
        m = opt.cyclelength;
    else
        m = 50;
    end

    if isfield(opt,'recyclealg')
        recyclealg = opt.recyclealg;
    else
        recyclealg = 'harmvecs';
    end
    
    existM2 = isfield(opt,'M2'); 
    if ~(existM2)
       M2 = [];
    else
        M2 = opt.M2;
    end
    
    existM1 = isfield(opt,'M1'); 
    if ~(existM1)
       M1 = [];
    else
        M1 = opt.M1;
    end
    
    if isfield(opt,'k')
        k = opt.k;
    else
        k = 10;
    end
    
    if isfield(opt,'isPrintStatus')
       isPrintStatus = opt.isPrintStatus;
    else
        isPrintStatus = false;
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % U declared as persistent to  %
    % avoid multiple copies        %
    % in memory due to recursion.  %
    % opt.U is set to empty to     %
    % to avoid it being passed in  %
    % the recursion call           %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    persistent U;
    %isInitSpace = ~isempty(U);
    
    if isfield(opt,'U') && ~isempty(opt.U)
        U = opt.U;
        %isInitSpace = true;
        opt.U = [];
    else
        U = [];
    end
    
    if isfield(opt,'isComplex')
        isComplex = opt.isComplex;
    else
        isComplex = false;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Logical array with true/false values       %
    % for various possible opt outputs           %    
    % so we can test if we need to initialize    %
    % opt                                        %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% 
    isOpt = true(0);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Check to see if we                 %
    % should output the updated U        %
    % and the option is turned off       %
    % for all subsequent recursive calls %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if isfield(opt,'isOutU')
        isOutU = opt.isOutU;
        isOpt = [isOpt isOutU];
        opt.isOutU = false;
    else
        isOutU = false;
    end
    
    if isfield(opt,'isOutNMV')
        isOutNMV = opt.isOutNMV;
        isOpt = [isOpt isOutNMV];
    else
        isOutNMV = true;
    end

    if sum(isOpt) > 0
        out = struct();
        %put initializations here
    else
        out = [];
    end

    %%%%%%%%%%%%%
    % Main Body %
    %%%%%%%%%%%%%
    [x, ~, resvec, ~, out_temp.nmv, U, ip] = rgmres(A, b, tol, m, k, nrestarts, M1, M2, x0, U, isComplex, recyclealg, isPrintStatus);
   
    if isOutNMV
        out.nmv = out_temp.nmv;
    end
    
    if isOutU
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        % Outermost function call %
        % Output U for future use %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%
        out.U = U;
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        % U is only persistent for the      %
        % purposes of recursively           %
        % calling the function.  Its value  %
        % should not persist between calls  %
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        U = [];
    end
    
    if isPrintStatus
        print_status('clear');
    end

    out.ip = ip;

end

function [x, relres, resvec, iter, nmv, U, ip] = rgmres(A, b, tol, m, k, nrestarts, M1, M2, x0, U, isComplex, recyclealg, isPrintStatus)

    ip = 0;
    n = length(b);

    isOpProc = isa(A,'function_handle');
    
    if ~exist('b','var') || isempty(b)
        error('%s%s','The right hand side is empty or not defined .',... 
                     'It is required to determine system dimeion.');
    end

    if ~exist('tol','var') || isempty(tol)
        tol = 1e-5;
    end

    if ~exist('m','var') || isempty(m)
        m = 100;
    end

    if ~exist('nrestarts','var') || isempty(nrestarts)
        nrestarts = 5;
    end
    
    existM1 = ~( ~exist('M1','var') || isempty(M1) );
    if ~(existM1)
       M1 = [];
       isLPrecProc = false;
    else
       isLPrecProc = isa(M1,'function_handle');
    end
    
    existM2 = ~( ~exist('M2','var') || isempty(M2) );
    if ~(existM2)
       M2 = [];
       isRPrecProc = false;
    else
       isRPrecProc = isa(M2,'function_handle');
    end

    if ~exist('x0','var') || isempty(x0)
        x0 = zeros(n,1);
    end
    
    existU = ~( ~exist('U','var') || isempty(U) );
   
    % initialize solution vector
    t = zeros(size(x0));
    
    % allocate space for U
    if ~existU
        % k is the requestred number of approximate eigenvectors to keep from one
        % cycle to the next This subspace may be built up over several linear
        % solves. As we use only a real representation, there are cases where keff
        % may take on the value k+1. keff only changes within getHarmVecs or
        % getHarmVecs2
        
        keff = 0;
        U = zeros(n,k);
    end
    
    C = zeros(n,k);

  
    % Calculate initial residuals
    if isOpProc
        r=b-A(x0);
    else
        r=b-A*x0; 
    end

   % initialize matvec count
    nmv = 1;
    
    if existM1
        if isLPrecProc
            r = M1(r);
        else
            r = M1\r;
        end
    end

    % Calculate initial preconditioned residual norm
    resvec = norm(r);
    ip = ip + 1;
    
    if existM1
       if isLPrecProc
           bnorm = norm(M1(b));
           ip = ip + 1;
       else
           bnorm = norm(M1\b);
           ip = ip + 1;
       end
    else
        bnorm = norm(b);
        ip = ip + 1;
    end
    

    %%%%%%%%%%%%%%%%%%%%%%%%%  Initialize U  %%%%%%%%%%%%%%%%%%%%%%%%%
    if existU
       cycle = 0;

       % Set size of effective k
       keff = size(U,2);
        
       %grow U to correct size for allocation purposes
       if keff < k
           U(:,keff+1:k) = zeros(n,k-keff);
       end

       % C = A * U (with preconditioning) We can frequently represent A_new =
       % A_old + deltaA. Note that
       % A_new * U = A_old*U + delta_A*U = C_old   + delta_A*U
       % where we already have C_old. Computing deltaA*U is generally much less
       % expensive than computing A_new*U, so we do not record these keff
       % matvecs
       %
       % ZU is stored for projection of the shifted systems when a
       % right preconditioner exists
       if(existM2)
          if isRPrecProc
              C(:,1:keff) = M2(U(:,1:keff));
          else
              C(:,1:keff) = M2 \ U(:,1:keff);
          end
       else
          C(:,1:keff) = U(:,1:keff);
       end
       if isOpProc
           C(:,1:keff) = A(C(:,1:keff));
       else
           C(:,1:keff) = A*C(:,1:keff);
       end

      %  nmv = nmv + keff; ? (from the above explanation we dont count)

       if(existM1)
          if isLPrecProc
              C(:,1:keff) = M1(U(:,1:keff));
          else
              C(:,1:keff) = M1 \ U(:,1:keff);
          end
       end
       % Orthonormalize C and adjust U accordingly so that C = A*U
       [C(:,1:keff),F] = qr(C(:,1:keff),0);
       U(:,1:keff) = U(:,1:keff) / F;

       % Update solution and residual norm for base system
       t = t + U(:,1:keff)*(C(:,1:keff)'*r);
       r = r - C(:,1:keff)*(C(:,1:keff)'*r);
       resvec(1) = norm(r);

       % Test for convergence: Was solution in recycle space?
       if (resvec(1)/bnorm < tol)
          % Assign all (unassigned) outgoing values and return
          if(existM2)
             if isRPrecProc
                 x = x0 + M2(t);
             else
                 x = x0 + M2 \ t;
             end
          else
             x = x0 + t;
          end 
          %flag = 0;
          relres = resvec(1) / bnorm;
          iter = [cycle 0];
          nmv = nmv - 1;
          U = U(:,1:keff);
          return
       end   
    else
       cycle = 1;
       % I have no subspace to recycle from a previous call to the solver
       % Perform m GMRES iterations to produce Krylov subspace of dimension m
       [t,r,p,resvec_inner,V,H, cycle_ip] = gmres_cycle(A,t,r,m,M1,M2,tol*bnorm,isOpProc,isLPrecProc, isRPrecProc, isPrintStatus);   
       % Record residual norms and increment matvec count
       resvec(2:p+1) = resvec_inner;
       nmv = nmv + p;
       ip = ip + cycle_ip;

       if k >= p % keep everything
         % form new U (the subspace to recycle)       
         U(:,1:p) = V(:,1:p);
         keff = p;
       else % downselect to k vectors
         if (strcmp(recyclealg,'harmvecs'))% || strcmp(recyclealg,'shr'))
            [P,keff] = getHarmVecsKryl(p,k,H,isComplex);
            % Form U (the subspace to recycle)
            U(:,1:keff) = V(:,1:p) * P;
         elseif(strcmp(recyclealg,'ritzvecs'))% || strcmp(recyclealg,'shr'))
            [P,keff] = getHarmVecsKryl(p,k,H,isComplex);
            % Form U (the subspace to recycle)
            U(:,1:keff) = V(:,1:p) * P;
         end
       end
       %disp(sprintf('%i vectors in recycle space',keff))   

       % If p < m, early convergence of GMRES
       if p < m
         % Assign all (unassigned) outgoing values and return
         if(existM2)
            if isRPrecProc
                x = x0 + M2(t);
            else
                x = x0 + M2 \ t;
            end
         else
            x = x0 + t;
         end 
         %flag = 0;      
         relres = resvec(nmv) / bnorm;
         iter = [cycle p];      
         nmv = nmv - 1;  
         U = U(:,1:keff);
         return
       end

       % Continuing to another cycle; Form orthonormalized C and adjust U accordingly so that C = A*U
       if (k >= p) % keep everything
         [QQ,F] = qr(H,0);
         C(:,1:keff) = V * QQ;
         U(:,1:keff) = U(:,1:keff) / F;
       else % not enough room for all Krylov vectors
         if (strcmp(recyclealg,'harmvecs') || strcmp(recyclealg,'ritzvecs'))      
            [QQ,F] = qr(H*P,0);
            C(:,1:keff) = V * QQ;
            U(:,1:keff) = U(:,1:keff) / F;
         else
             error('No other recycle algorithms have been implemented yet');
         end
       end
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%  Main Body of Solver.  %%%%%%%%%%%%%%%%%%%%%%%%%
    while(cycle <= nrestarts-1)
       % Do m iterations of GMRES
       [t,r,p,resvec_inner,V,~,H,B, cycle_ip] = rgmres_cycle(A,t,r,m,M1,M2,C(:,1:keff),U(:,1:keff),tol*bnorm,isOpProc, isLPrecProc, isRPrecProc, isPrintStatus);
       cycle = cycle + 1;      
       ip = ip + cycle_ip;
       % Record residual norms and increment matvec count   
       resvec(nmv+1:nmv+p) = resvec_inner;
       nmv = nmv + p;

       if (k-keff) >= p % keep everything
         % Keep all Krylov vectors, update keff
         U(:,keff+1:keff+p) = V(:,1:p);
         keff_old = keff;
         keff = keff_old + p;     
       else % downselect to k vectors
        % Rescale U and store the inverses of the norms of its columns in the diagonals of D.
        clear d;
        d = zeros(1,keff);
        for i = 1:keff
            d(i) = norm(U(:,i));
            U(:,i) = U(:,i) / d(i);
        end
        D = diag(1 ./ d); 
        % Form large H
        H2 = zeros(p+keff+1,p+keff);
        H2(1:keff,1:keff)                 = D;
        H2(1:keff,keff+1:p+keff)          = B;
        H2(keff+1:p+keff+1,keff+1:p+keff) = H;
        % Calculate Harmonic Ritz vectors
        keff_old = keff;
            if (strcmp(recyclealg,'harmvecs'))% || strcmp(recyclealg,'shr'))
                [P,keff,ip_harm_ritz] = getHarmVecsAug(p,k,keff_old,H2,V,U(:,1:keff_old),C(:,1:keff_old),isComplex); 
                % Form new U
                U = U(:,1:keff_old) * P(1:keff_old,:) + V(:,1:p) * P(keff_old+1:keff_old+p,:);
            elseif strcmp(recyclealg,'ritzvecs')
                
            end
            ip = ip + ip_harm_ritz;
       end    

       % If p < m, early convergence of GMRES
       if p < m
          % Assign all (unassigned) outgoing values and return
          if(existM2)
             if isRPrecProc
                 x = x0 + M2(t);
             else
                 x = x0 + M2 \ t;
             end
          else
             x = x0 + t;
          end      
          relres = resvec(p+1) / bnorm;
          nmv = nmv - 1;
          flag = 0;
          iter = [cycle p];
          U = U(:,1:keff);
          return
       end

       % Continuing to another cycle; Form orthonormalized C and adjust U accordingly so that C = A*U
       if (k-keff_old) >= p % keep everything
         % Form large H
         H2 = zeros(p+keff_old+1,p+keff_old);
         H2(1:keff_old,1:keff_old)                         = eye(keff_old);
         H2(1:keff_old,keff_old+1:p+keff_old)              = B;
         H2(keff_old+1:p+keff_old+1,keff_old+1:p+keff_old) = H;
         [Q,F] = qr(H2,0);
         C = C(:,1:keff_old) * Q(1:keff_old,:) + V * Q(keff_old+1:keff_old+p+1,:);%[C V] * Q;
         U(:,1:keff) = U(:,1:keff) / F;
       else % not enough room for all Krylov vectors
         if (strcmp(recyclealg,'harmvecs'))% || strcmp(recyclealg,'shr')) 
           % H2, U already updated 
           [Q,F] = qr(H2*P,0);
           C =  C(:,1:keff_old) * Q(1:keff_old,:) + V * Q(keff_old+1:keff_old+p+1,:);%[C V] * Q;
           U(:,1:keff) = U(:,1:keff) / F;
         else
           error('No other recycle algorithms have been implemented yet');
         end
       end
    end

    % Exceeded nrestarts iterations
    %flag = 1;

    % Calculate final solution and residual.
    if(existM2)
       if isRPrecProc
           x = x0 + M2(t);
       else
           x = x0 + M2 \ t;
       end
    else
       x = x0 + t;
    end

    % Calculate relative residual.
    relres = resvec(nmv) / bnorm;

    % Correct matvec count
    nmv = nmv - 1;

    iter = [cycle p];
    U = U(:,1:keff);
end

function e = euclbasis(j,n)
    e = zeros(n,1);
    e(j) = 1;
end

function x = UTSolve(R, y, n)

    x = zeros(n,1);

    x(n) = y(n) / R(n,n);
    for i = n-1:-1:1,
       x(i) = y(i);
       for j = i+1:n,
          x(i) = x(i) - R(i,j) * x(j);
       end
       x(i) = x(i) / R(i,i);
    end
end

function [x,r,j,resvec,V,G,ip] = gmres_cycle(A,x,r,m,M1,M2,tol,isOpProc, isLPrecProc, isRPrecProc, isPrintStatus)

    ip = 0;
    existM2 = ~isempty(M2);
    existM1 = ~isempty(M1);
    
    V = zeros(length(x),m+1);
    %Z = zeros(length(x),m);
    H = zeros(m+1,m);
    G = zeros(m+1,m);
    resvec = zeros(m,1);
    s = zeros(m,1);
    c = zeros(m,1);

    % Initialize V
    V(:,1) = r / norm(r);
    ip = ip + 1;

    % Initialize rhs for least-squares system
    rhs = zeros(m+1,1);
    rhs(1) = norm(r);

    % Do m steps of GMRES
    for j = 1:m
        if existM2
            if isRPrecProc
                V(:,j+1) = M2(V(:,j));
            else
                V(:,j+1) = M2 \ V(:,j);
            end
            %Z(:,j) = V(:,j+1);
        else
            V(:,j+1) = V(:,j);
        end
        
        if isOpProc
            V(:,j+1) = A(V(:,j+1));
        else
            V(:,j+1) = A*V(:,j+1);
        end
        if existM1
            if isLPrecProc
                V(:,j+1) = M1(V(:,j));
            else
                V(:,j+1) = M1 \ V(:,j);
            end
            %Z(:,j) = V(:,j+1);
        end
        
        if isPrintStatus
            print_status;
        end
        % Orthogonalize on V
        for i = 1:j
            H(i,j) = V(:,i)' * V(:,j+1);
            ip = ip + 1;
            G(i,j) = H(i,j);
            V(:,j+1) = V(:,j+1) - H(i,j)*V(:,i);
        end
        H(j+1,j) = norm(V(:,j+1));
        ip = ip + 1;
        G(j+1,j) = H(j+1,j);

        if (H(j+1,j) ~= 0.0)
            V(:,j+1) = V(:,j+1) / H(j+1,j);
        end

        % Perform plane rotations on new column
        for i = 1:j-1,
            h1 = H(i,j);
            h2 = H(i+1,j);
            H(i,j)   = conj(c(i)) * h1 + s(i) * h2;
            H(i+1,j) = -s(i) * h1 + c(i) * h2;
        end

        % Calculate new plane rotation      
        y1 = H(j,j);
        y2 = H(j+1,j);
        rh1 = rhs(j);
        rh2 = rhs(j+1);
        if y2 == 0.0
            c(j) = y1/conj(y1);
            s(j) = 0.0;
        elseif abs(y2) > abs(y1)
            h1 = (y1/y2);
            s(j) = 1.0 / sqrt(1 + abs(h1) * abs(h1));
            c(j) = s(j) * h1;
        else
            h1 = (y2/abs(y1));
            c(j) = y1 / abs(y1) / sqrt(1 + h1 * h1);
            s(j) = h1 / sqrt(1 + h1 * h1);
        end
        h1       = y1;
        h2       = y2;
        H(j,j)   = conj(c(j)) * h1 + s(j) * h2;
        H(j+1,j) = -s(j) * h1 + c(j) * h2;
        rhs(j)   = conj(c(j)) * rh1;      
        rhs(j+1) = -s(j) * rh1;      

        resvec(j) = abs(rhs(j+1));

        if (resvec(j) < tol) 
            break;
        end

    end

    % Solve least squares system
    y = UTSolve(H(1:j+1,1:j),rhs,j);

    % Calculate solution
    x = x + V(:,1:j) * y;
    e1 = euclbasis(1,j+1);
    % Calculate residual
    x0 = zeros(j+1,1);
    x0(j+1) = -rhs(j+1);
    for i = j:-1:1
        x0(i)   = x0(i+1) * -s(i);
        x0(i+1) = x0(i+1) * conj(c(i));
    end
    x0(1) = x0(1) + norm(r);
    r = r - V(:,1:j+1)*x0;
    
    V = V(:,1:j+1);
    %Z = Z(:,1:j);   
    G = G(1:j+1,1:j);
    resvec = resvec(1:j);
end

function [x,r,j,resvec,V,Z,G,B, ip] = rgmres_cycle(A,x,r,m,M1,M2,C,U,tol,isOpProc, isLPrecProc,isRPrecProc, isPrintStatus)

    ip = 0;

    existM2 = ~isempty(M2);
    existM1 = ~isempty(M1);

    gamma = norm(r);
    V = zeros(length(x),m+1);
    Z = zeros(length(x),m+1);
    B = zeros(size(U,2),m);
    H = zeros(m+1,m);
    G = zeros(m+1,m);
    c = zeros(1,m);
    s = zeros(1,m);
    resvec = zeros(m,1);

    % Initialize V
    V(:,1) = r / gamma;

    % Initialize rhs for least-squares system
    rhs = zeros(m+1,1);
    rhs(1) = gamma;

    % Do m steps of GMRES
    for j = 1:m
        if(existM2)
            if isRPrecProc
                V(:,j+1) = M2(V(:,j));
            else
                V(:,j+1) = M2 \ V(:,j);
            end
            Z(:,j) = V(:,j+1);
        else
            V(:,j+1) = V(:,j);
        end
        if isOpProc
            V(:,j+1) = A(V(:,j+1));
        else
            V(:,j+1) = A*V(:,j+1);
        end
        if(existM1)
            if isLPrecProc
                V(:,j+1) = M1(V(:,j));
            else
                V(:,j+1) = M1 \ V(:,j);
            end
            Z(:,j) = V(:,j+1);
        end
        
        if isPrintStatus
            print_status;
        end

        % Orthogonalize on C
        for i = 1:size(C,2)
            B(i,j)   = C(:,i)'*V(:,j+1);
            ip = ip + 1;
            V(:,j+1) = V(:,j+1) - B(i,j)*C(:,i);
            ip = ip + 1;
        end

        % Orthogonalize on V
        for i = 1:j
            H(i,j) = V(:,i)' * V(:,j+1);
            ip = ip + 1;
            G(i,j) = H(i,j);      
            V(:,j+1) = V(:,j+1) - H(i,j)*V(:,i);
        end
        H(j+1,j) = norm(V(:,j+1));
        ip = ip + 1;
        G(j+1,j) = H(j+1,j);   

        if (H(j+1,j) ~= 0.0) 
            V(:,j+1) = V(:,j+1) / H(j+1,j);
        end

        % Perform plane rotations on new column
        for i = 1:j-1,
            h1 = H(i,j);
            h2 = H(i+1,j);
            H(i,j)   = conj(c(i)) * h1 + s(i) * h2;
            H(i+1,j) = -s(i) * h1 + c(i) * h2;
        end

        % Calculate new plane rotation      
        y1 = H(j,j);
        y2 = H(j+1,j);
        rh1 = rhs(j);
        rh2 = rhs(j+1);
        if y2 == 0.0
            c(j) = y1/conj(y1);
            s(j) = 0.0;
        elseif abs(y2) > abs(y1)
            h1 = (y1/y2);
            s(j) = 1.0 / sqrt(1 + abs(h1) * abs(h1));
            c(j) = s(j) * h1;
        else
            h1 = (y2/abs(y1));
            c(j) = y1 / abs(y1) / sqrt(1 + h1 * h1);
            s(j) = h1 / sqrt(1 + h1 * h1);
        end
        h1       = y1;
        h2       = y2;
        H(j,j)   = conj(c(j)) * h1 + s(j) * h2;
        H(j+1,j) = -s(j) * h1 + c(j) * h2;
        rhs(j)   = conj(c(j)) * rh1;      
        rhs(j+1) = -s(j) * rh1;      

        resvec(j) = abs(rhs(j+1));

        if (resvec(j) < tol) 
            break;
        end
    
    end

    % Solve least squares system
    y = UTSolve(H(1:j+1,1:j),rhs,j);

    % Calculate solution
    x = x + V(:,1:j) * y;
    x = x - U*(B(1:size(U,2),1:j)*y);
    
    e1 = euclbasis(1,j+1);

    % Calculate residual
    x0 = zeros(j+1,1);
    x0(j+1) = -rhs(j+1);
    for i = j:-1:1
        x0(i)   = x0(i+1) * -s(i);
        x0(i+1) = x0(i+1) * conj(c(i));
    end
    x0(1) = x0(1) + norm(r);
    r = r - V(:,1:j+1)*x0;
    % Compute Shifted System's Solutions
    z = gamma*e1 - x0;
    
    V = V(:,1:j+1);
    Z = Z(:,1:j);
    G = G(1:j+1,1:j);
    B = B(1:size(U,2),1:j);
    resvec = resvec(1:j);

end

function [PP,keff] = getHarmVecsKryl(m,k,H, isComplex)
    % Build matrix for eigenvalue problem
    harmRitzMat = H(1:m,:)' \ speye(m);
    harmRitzMat(1:m,1:m-1) = 0;
    harmRitzMat = H(1:m,:) + H(m+1,m)^2 * harmRitzMat;

    % Compute k smallest harmonic Ritz pairs
    [harmVecs, harmVals] = eig(harmRitzMat);
    harmVals = diag(harmVals);

    % Construct magnitude of each harmonic Ritz value
    w = abs(harmVals);
    [~,iperm] = sort(w);  

    % Select k smallest eigenvectors
    % Optionally store k+1 vectors to capture complex conjugate pair
    idx = 1;
    while(idx <= k)
      if (isreal(harmVals(iperm(idx)))) || isComplex
        PP(:,idx) = harmVecs(:,iperm(idx));
        idx = idx + 1;    
      else
        PP(:,idx) = real(harmVecs(:,iperm(idx)));
        PP(:,idx+1) = imag(harmVecs(:,iperm(idx)));
        idx = idx + 2;      
      end
    end

    % Return number of vectors selected
    keff = idx-1;
end

function [PP,keff,ip] = getHarmVecsAug(m,k,keff,G,V,U,C, isComplex)
   
    ip = 0;
  
    B = G' * G;
   
    % A = | C'*U        0 |
    %     | V_{m+1}'*U  I |
    A = zeros(m-k+keff+1,m-k+keff);
    A(1:keff,1:keff) = C' * U;
    A(keff+1:m+keff+1,1:keff) = V' * U;
    ip = ip + k;
    A(keff+1:m+keff,keff+1:m+keff) = eye(m);
    A = G' * A;

    % Compute k smallest harmonic Ritz pairs
    [harmVecs, harmVals] = eig(A,B);
    harmVals = diag(harmVals);

    % Construct magnitude of each harmonic Ritz value
    w = abs(harmVals);
    [~,iperm] = sort(w,1,'descend');  

    % k smallest harmonic ritz values
    % Actually, k largest of (1./harmonic Ritz value)
    % Optionally store k+1 vectors to capture complex conjugate pair
    PP = zeros(m+keff,k+1);
    idx = 1;
    while(idx <= k)
      if (isreal(harmVals(iperm(idx)))) || isComplex
        try PP(:,idx) = harmVecs(:,iperm(idx)); catch er; keyboard; end
        idx = idx + 1;    
      else
        PP(:,idx) = real(harmVecs(:,iperm(idx)));
        PP(:,idx+1) = imag(harmVecs(:,iperm(idx)));
        idx = idx + 2;      
      end
    end

    % Return number of vectors selected
    keff = idx-1;
    if keff == k
        PP = PP(:,1:k);
    end
end

function print_status(special_command)
%
% Prints a status symbol at the command line, overwriting the first
% character of the line to let the user know the calculation is running.
% Uses persistent variables to keep track of whether it needs to backspace
% before printing as well as to keep track of which symbol to use.
%
% special_command    Pass along a command to clear persistent variables
%
% Kirk Soodhalter
% 24-Jan-2014 13:24:02
% kirk@math.soodhalter.com

    persistent doBackspace charNum;

    if isempty(doBackspace)
        doBackspace = false; 
    end

    if isempty(charNum)
        charNum = 1;
    end

    if ~exist('special_command','var')
        special_command = '';
    end

    if doBackspace
        fprintf('\b');
    end

    if strcmp(special_command,'clear')
        doBackspace = false;
        charNum = 1;
    else
        fprintf(status_symbol(charNum));

        charNum = charNum + 1;
        if charNum == 5
            charNum = 1;
        end

        doBackspace = true;
    end
end

function symbol = status_symbol(k)
% symbol = status_symbol(k)
%
% For printing a status symbol on the command line, so it is clear the
% calculation is moving.
%
% DATA DICTIONARY
% INPUTS:
%   k     Can be 1,2,3,4 which returns, respectively, '|', '/', '-', 
%         '\'.  We note that char(8211) is the long dash character.
% OUTPUTS:
%   symbol   Returns the appropriate character.
%
% Kirk Soodhalter
% 24-Jan-2014 13:11:18
% kirk@math.soodhalter.com

    switch k
        case 1
            symbol = '|';
        case 2
            symbol = '/';
        case 3
            symbol = '-';
        case 4
            symbol = '\\'; %escape code to get actual backslash
        otherwise
            error('A value of k=%d was entered, but k can only be 1, 2, 3, or 4',k);
    end
end
