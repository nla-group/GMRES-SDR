function [x,out] = gmres_dr(A,b,param)
% GMRES-DR with harmonic restarting as described by Morgan (2002)

n = size(b,1);

if isnumeric(A)
    A = @(v) A*v;
end

if ~isfield(param,'x0')
    x = zeros(n,1);  % Assuming initial zero guess
    r = b;
else
    x = param.x0;
    r = b - A(x);
end

if ~isfield(param,'max_it')
    param.max_it = 50;
end

if ~isfield(param,'max_restarts')
    param.max_restarts = min(ceil(n/param.max_it), 10);
end

if ~isfield(param,'k')
    param.k = min(10, param.max_it);
end

if ~isfield(param,'tol')
    param.tol = 1e-6;
end

k = param.k; 
V = zeros(n,k + param.max_it + 1);
bet = norm(r);
V(:,1) = r/bet; 
Vr = bet; % V'*r without trailing zeros
H = []; out.ip = 0; out.mv = 0;
keep = 0; % initially no harmonic Ritz pairs
out.resvec = [bet];
for restart = 1:param.max_restarts

    mk = param.max_it + keep;
    Q = []; R = []; % initial QR of H

    % inner loop
    for j = keep+1:mk
        w = A(V(:,j));
        out.mv = out.mv + 1;
        for i = 1:j
            H(i,j) = V(:,i)'*w;
            w = w - V(:,i)*H(i,j);
            out.ip = out.ip + 1;
        end
        H(j+1,j) = norm(w);
        out.ip = out.ip + 1;
        V(:,j+1) = w/H(j+1,j);

        % estimate residual (and compute approximation)
        %norm(A*V(:,1:mk) - V*H)
        %c = bet*eye(m+1,1); % only in first cycle

        %c = V(:,1:j+1)'*r; % TODO: this can be done directly!!!
        c = Vr;
        c(j+1,1) = 0;
        
        %d = pinv(H(1:j+1,1:j))*c;
        if ~isempty(Q)
            Q(j+1,1) = 0; % extend Q
        end
        [Q,R] = qrupdate_gs(H(1:j+1,1:j),Q,R);
        d = R\(Q'*c);
        
        rc = c - H(1:j+1,1:j)*d;
        res = norm(rc);
        out.resvec = [ out.resvec, res ];

        if j == mk || res < param.tol
            r = V(:,1:j+1)*rc;
            x = x + V(:,1:j)*d;
         %   fprintf('  ||r|| = %5.2e\n',norm(r))
        end
        if res < param.tol
            out.U = V(:,1:keep);
            return
        end
    end

    % harmonic ritz
    emk = zeros(mk,1); emk(mk) = 1;
    Ht = H(1:mk,1:mk) + H(mk+1,mk)^2*(H(1:mk,1:mk)'\emk)*emk';

    % follow Morgan's paper here but using Schur vectors
    %[X,D] = eig(Ht);

    [X,T] = schur(Ht);
    ritz = ordeig(T);
    [~,ind] = sort(abs(ritz),'ascend');
    select = false(length(ritz),1);
    select(ind(1:k)) = 1;
    [X,T] = ordschur(X,T,select);  % H = X*T*X'

    if k>0 && T(k+1,k)~=0   % don't tear apart 2x2 diagonal blocks of Schur factor
        keep = k+1;
    else
        keep = k;
    end

   % disp(['  recycling subspace dimension = ' num2str(keep)])

    Pk = X(:,1:keep);
    Pkp1 = [ Pk ; zeros(1,keep) ];
    Pkp1 = [ Pkp1 , rc ];
    [Pkp1,~] = qr(Pkp1,0);
    Pkp1(1:mk,1:keep) = Pk; % prevent sign changes

    H = Pkp1'*H(1:mk+1,1:mk)*Pk;
    V = V(:,1:mk+1)*Pkp1;

    %Vr = V'*r; % coefficients of residual in new basis
    Vr = Pkp1'*rc;

    %norm(A*V(:,1:keep) - V*H)/norm(H)
end

