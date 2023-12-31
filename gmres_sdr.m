function [x,out] = gmres_sdr(A,b,param)
% [x,out] = GMRES_SDR(A,b,param) 
%
% A function which solves a linear system Ax = b using GMRES
% with sketching and deflated resarting. 
% The recycling subspace is updated using (harmonic) Ritz extraction.
%
% INPUT: A     matrix or function handle
%        b     right-hand side vector
%        param input struct with the following fields
%
%        param.x0           initial guess
%        param.max_it       maximum number of inner Arnoldi iterations
%        param.max_restarts maximum number of outer restarts
%        param.tol          absolute target residual norm
%        param.U            basis of the recycling subspace
%        param.SU, SAU      sketched basis of recycling subspace
%        param.k            target dimension of recycling subspace
%        param.t            Arnoldi truncation parameter
%        param.hS           handle to subspace embedding operator
%        param.s            subspace embedding dimension
%        param.verbose      control verbosity, can be 0,1,2
%
% OUTPUT: out An output struct with the following fields
%         out.residuals: Vector containing residual norms for each cycle of
%                        of final system
%         out.U          updated recycling subspace
%         out.SU, SAU    sketches of updated recycling subspace
%         out.mv         number of matrix-vector products
%         out.ip         number of inner-products
%         out.s          number of sketched vectors

% Initialize counter variables
mv = 0;     % Number of matrix vector products
sv = 0;     % Number of vector sketches

if isnumeric(A)
    A = @(v) A*v;
end

n = size(b,1); 

if ~isfield(param,'verbose')
    param.verbose = 1;
end

if ~isfield(param,'U')
    param.U = []; param.SU = []; param.SAU = [];
end

if ~isfield(param,'max_it')
    param.max_it = 50;
end

if ~isfield(param,'max_restarts')
    param.max_restarts = min(ceil(n/param.max_it), 10);
end

if ~isfield(param,'tol')
    param.tol = 1e-6;
end

if ~isfield(param,'x0')
    x = zeros(n,1);  % Assuming initial zero guess
    r = b;
else
    x = param.x0;
    r = b - A(x);
end

if ~isfield(param,'ssa')
    param.ssa = 0;  % Sketch-and-select?
end

if ~isfield(param,'t')
    param.t = 2;  % truncation param
end

if ~isfield(param,'k')
    param.k = min(10, param.max_it);
end

if ~isfield(param,'s')
    param.s = min(n, 8*(param.max_it + param.k));
end

if param.verbose
    % please stop commenting these out! use verbose=0
    disp(['  using sketching dimension of s = ' num2str(param.s)])
end

if ~isfield(param,'reorth')
    param.reorth = 0;
end

if ~isfield(param,'hS') || isempty(param.hS)
    param.hS = srft(n,param.s);
end

if ~isfield(param, 'sketch_distortion')
    param.sketch_distortion = 1.4; % assumed upper bound on distortion factor
end

if ~isfield(param, 'ls_solve')
    param.ls_solve = 'mgs'; % how to solve the least squares problems
end

if ~isfield(param, 'svd_tol')
    param.svd_tol = 1e-15; % SVD stabilization for recycling subspace
end

if ~isfield(param, 'harmonic')
    param.harmonic = 1; % harmonic/standard Ritz for recycling subspace
end

if ~isfield(param, 'd')
    param.d = 1; % TODO: modify inner loop to deal properly with d>1 case
end

% Vector to store the true residual norm at end of each cycle
residuals = [norm(b)];
ip = 1;

% store number of inner iterations per restart cycle
iters = 0;
% store sketched residual at every inner iteration
sres = NaN; % initial sketch not available here 

for restart = 1:param.max_restarts

    % Call cycle of srgmres
    [ e, r, cycle_out ] = srgmres_cycle(A, r, param);

    % update solution approximation
    x = x + e;

    % Compute and store norm of residual
    residnorm = norm(r);
    residuals(restart+1) = residnorm;
    iters(restart+1) = cycle_out.m;

    sres = [ sres , cycle_out.sres ];

    %fprintf('  ||r|| = %5.2e\n',residnorm)

    % Increment counter variables appropriately
    mv = mv + cycle_out.mv;
    ip = ip + cycle_out.ip;
    sv = sv + cycle_out.sv;

    % Output recycling matrices for next cycle
    param.U = cycle_out.U; param.SU = cycle_out.SU;
    param.SAU = cycle_out.SAU;

    % Potentially increase bound on sketch distortion
    param.sketch_distortion = cycle_out.sketch_distortion;

    if norm(r) < param.tol
        break
    end

end

% Output relevant parameters
out = cycle_out;
out.mv = mv;
out.ip = ip;
out.sv = sv;
out.residuals = residuals;
out.iters = iters;
out.sres = sres;

end


function [e,r,out] = srgmres_cycle(A,r0,param)

max_it = param.max_it;
tol = param.tol;
hS = param.hS;
t = param.t;
U = param.U;
k = param.k;
d = param.d;

sketch_distortion = param.sketch_distortion;

% Reset count parameters for each new cycle
mv = 0;
ip = 0;
sv = 0;

if isempty(U)
    SW = [];
    SAW = [];
else
    % In the special case when the matrix does not change, 
    % we can re-use SU from previous problem,
    if param.pert == 0
        SW = param.SU;
        SAW = param.SAU;
        mv = mv + 0;
    else
        SW = param.SU;
        if isempty(U)
            SAW = [];
        else
            SAW = hS(A(U));
            mv = mv + size(U,2);
            sv = sv + size(U,2);
        end
    end
end

% Arnoldi for (A,b)
Sr = hS(r0);
sv = sv + 1;
if param.ssa
    nrm = norm(Sr);
else
    nrm = norm(r0);
    ip = ip + 1;
end
SV(:,1) = Sr/nrm;
V(:,1) = r0/nrm;

% NOTE: Interestingly, the vectors for which the distortion 
% norm(Sv)/norm(v) is largest away from 1, happen to be the
% residual vectors after each restart (at least with dct).
% What happens is that within a circle, norm(Sr)/norm(r) 
% typically starts to deviate more and more from 1 and 
% as the next cycle is restarted with a residual vector,
% it has large distortion. 
%
%nrmV = norm(V(:,1));
%nrmSV = norm(SV(:,1));

% Initialize QR factorization of SAW (recycling subspace)
if strcmp(param.ls_solve,'mgs') % modified GS
    [Q,R] = qr(SAW,0);
end
if strcmp(param.ls_solve,'hh') % Householder
    [W,R,QtSr] = qrupdate_hh(SAW,[],[],Sr);
end

d_it = 0; sres = []; 

for j = 1:max_it

    w = A(V(:,j));
    mv = mv + 1;

    if param.ssa == 0     % standard t-truncated Arnoldi
        for i = max(j-t+1,1):j
            H(i,j) = V(:,i)'*w;
            ip = ip + 1;
            w = w - V(:,i)*H(i,j);
        end
        H(j+1,j) = norm(w);
        ip = ip + 1;
        V(:,j+1) = w/H(j+1,j);
        SV(:,j+1) = hS(V(:,j+1));
        sv = sv + 1;
        % No need to sketch A*V since S*A*V = (S*V)*H
        %SAV = SV(:,1:j+1)*H(1:j+1,1:j);
        SAV(:,j) = SV(:,1:j+1)*H(1:j+1,j); 
    end

    if param.ssa == 1     % sketched t-truncated Arnoldi
        sw = hS(w); sv = sv + 1;
        SAV(:,j) = sw;

        % quasi-orthogonalise against U
        if size(param.U,2)>0
            coeffs = pinv(param.SU)*sw;
            w = w - param.U*coeffs;
            sw = sw - param.SU*coeffs;
        end
        
        % get coeffs with respect to previous t vectors
        ind = max(j-t+1,1):j;
        coeffs = SV(:,ind)'*sw;

        w = w - V(:,ind)*coeffs;
        %w = w - submatxmat(V,coeffs,min(ind),max(ind)); 

        sw = sw - SV(:,ind)*coeffs;
        nsw = norm(sw);
        SV(:,j+1) = sw/nsw; V(:,j+1) = w/nsw;
        H(ind,j) = coeffs; H(j+1,j) = nsw;
    end


    if param.ssa == 2     % sketch-and-select
        sw = hS(w); sv = sv + 1;
        SAV(:,j) = sw;
        % the following two lines perform the select operation
        coeffs = pinv(SV(:,1:j))*sw;
        [coeffs,ind] = maxk(abs(coeffs),t);
        w = w - V(:,ind)*coeffs;
        sw = sw - SV(:,ind)*coeffs;
        nsw = norm(sw);
        SV(:,j+1) = sw/nsw; V(:,j+1) = w/nsw;
        H(ind,j) = coeffs; H(j+1,j) = nsw;
    end

       
    % Every d iterations, compute the sketched residual 
    % If sres is small enough, compute full residual
    % If this is small enough, break the inner loop
    if rem(j,d) == 0 || j == max_it 

        d_it = d_it + 1;

        % TODO: Both could be updated column-wise
        SW = [ param.SU, SV(:,1:j) ];
        SAW = [ param.SAU, SAV(:,1:j) ];

        if ~isempty(U)
            %keyboard
        end
    
        % Incrementally extend QR factorization and get LS coeffs
        if strcmp(param.ls_solve,'mgs')
            [Q,R] = qrupdate_gs(SAW,Q,R);
            y = R\(Q'*Sr);
        end
        if strcmp(param.ls_solve,'hh')
            [W,R,QtSr] = qrupdate_hh(SAW,W,R,QtSr);
            y = triu(R)\(QtSr);
        end
        if strcmp(param.ls_solve,'pinv')
            y = pinv(SAW)*Sr;
        end
        if strcmp(param.ls_solve,'\')
            y = SAW\Sr;
        end

        % Compute residual estimate (without forming full approximation)
        sres(d_it) = norm(Sr - SAW*y);

        % If the residual estimate is small enough (or we reached the max
        % number of iterations), then we form the full approximation
        % correction (without explicitly forming [U V(:,1:j)])
        if sres(d_it) < tol/sketch_distortion || j == max_it
            if size(U,2) > 0
                e = U*y(1:size(U,2),1) + V(:,1:j)*y(size(U,2)+1:end,1);
            else
                e = V(:,1:j)*y(size(U,2)+1:end,1);
            end

            % Compute true residual
            r = r0 - A(e);
            mv = mv + 1;

            nrmr = norm(r);
            ip = ip + 1;

            % potentially increase sketch_distortion
            if nrmr/sres(d_it) > sketch_distortion
                sketch_distortion = nrmr/sres(d_it);
                if param.verbose >= 1
                    % please stop commenting these out! use verbose=0
                    disp(['  sketch distortion increased to ' num2str(sketch_distortion)])
                end
            end
            
            if nrmr < tol || j == max_it
                break
            end

        end
    end
end

% Compute economic SVD of SW or SAW
if param.harmonic
    [Lfull,Sigfull,Jfull] = svd(SAW,'econ');  % harmonic
else    
    [Lfull,Sigfull,Jfull] = svd(SW,'econ');   % non-harmonic
end

if param.verbose >= 2
    fprintf('  cond(SAU) = %4.1e\n', cond(param.SAU))
    fprintf('  cond(SV) = %4.1e\n', cond(SV(:,1:j)))
    fprintf('  full subspace condition number = %4.1e\n', Sigfull(1,1)/Sigfull(end,end))
end

% Truncate SVD
ell = find(diag(Sigfull) > param.svd_tol*Sigfull(1,1), 1, 'last');
k = min(ell,k);
L = Lfull(:,1:ell);
Sig = Sigfull(1:ell,1:ell);
J = Jfull(:,1:ell);
if param.harmonic
    HH = L'*SW*J;   % harmonic
else
    HH = L'*SAW*J;  % non-harmonic
end

% update augmentation space using QZ
if isreal(HH) && isreal(Sig)
    [AA, BB, Q, Z] = qz(HH,Sig,'real'); % Q*A*Z = AA, Q*B*Z = BB
else
    [AA, BB, Q, Z] = qz(HH,Sig);
end
ritz = ordeig(AA,BB);
if param.harmonic
    [~,ind] = sort(abs(ritz),'descend');  % harmonic
else
    [~,ind] = sort(abs(ritz),'ascend');   % non-harmonic
end

select = false(length(ritz),1);
select(ind(1:k)) = 1;
[AA,BB,~,Z] = ordqz(AA,BB,Q,Z,select);
if k>0 && k<size(AA,1) && (AA(k+1,k)~=0 || BB(k+1,k)~=0)  % don't tear apart 2x2 diagonal blocks
    keep = k+1;
else
    keep = k;
end

if param.verbose >= 2
    disp(['  recycling subspace dimension k = ' num2str(keep)])
end

% cheap update of recycling subspace without explicitly constructing [U V(:,1:j)]
JZ = J*Z(:,1:keep);
if size(U,2) > 0
    out.U = U*JZ(1:size(U,2),:) + V(:,1:j)*JZ(size(U,2)+1:end,:);
else
    out.U = V(:,1:j)*JZ(size(U,2)+1:end,:);
end

out.SU = SW*JZ;
out.SAU = SAW*JZ;
out.hS = hS;
out.k = keep;
out.m = j;
out.mv = mv;
out.ip = ip;
out.sv = sv;
out.sres = sres;
out.sketch_distortion = sketch_distortion;


end