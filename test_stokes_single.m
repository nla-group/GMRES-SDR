% TEST_STOKES_SINGLE
% Compare various methods for solving a single linear system of equations

clear all
close all
clc

% Download Stokes matrix from https://sparse.tamu.edu/VLSI/vas_stokes_1M
load('vas_stokes_1M.mat'); 
A = Problem.A; n = size(A,1);

% ILU preconditioner
[L,U] = ilu(A);
PA = @(x) U\(L\(A*x));

m = 100;          % max Arnoldi cycle length
nrestarts = 10;   % max number of restarts
k = 20;           % recycling subspace dimension
tol = 1e-6;       % residual tolerance
runs = 1;         % number of times to run experiment (for more robust timings)

% Create rhs and precondition
rng('default')
b = randn(n,1);
Pb = U\(L\b);
% we normalize Pb to compare different methods which may use relative or absolute residual
bet = norm(Pb);   
Pb = Pb/bet;     
b = b/norm(bet);

%% GMRES
disp('MATLAB gmres ******************************************************')
tic
for run = 1:runs
    gmres_matvec = 0;
    gmres_ip = 0;
    [x,flag,relres,iter,resvec] = gmres(PA,Pb,m,tol,nrestarts);
    gmres_matvec = gmres_matvec + length(resvec)-1;
    gmres_ip = gmres_ip + iter(1)*(m*(m+1)/2) + (iter(2)*(iter(2)+1))/2;
end
disp(['runtime = ' num2str(toc/runs)])
disp(['matvecs = ' num2str(gmres_matvec)])
disp(['ip = ', num2str(gmres_ip)]);
disp(['trueres = ' num2str(norm(b-A*x)/norm(b))])
semilogy(0:length(resvec)-1, resvec, 'k-'); hold on
xlabel('Number of iterations');
ylabel('Relative residual norm');
title("Stokes (single system)");
legend('GMRES'); shg

%% GCRO-DR
disp('gcrodr ************************************************************')
opts.k = k;
opts.cyclelength = m;
opts.nrestarts = nrestarts;
opts.tol = tol;
opts.isOutNMV = 1;
opts.isOutU = 1;
tic
for run = 1:runs
    gcrodr_matvec = 0;
    gcrodr_ip = 0;
    opts.U = [];
    [x,resvec,out2] = gcrodr(PA,Pb,opts);
    opts.U = out2.U;
    gcrodr_matvec = gcrodr_matvec + out2.nmv;
    gcrodr_ip = gcrodr_ip + out2.ip;
end
disp(['runtime = ' num2str(toc/runs)])
disp(['matvecs = ' num2str(gcrodr_matvec)])
disp(['ip = ' num2str(gcrodr_ip)])
disp(['trueres = ' num2str(norm(b-A*x)/norm(b))])
semilogy(0:length(resvec)-1,resvec, '-.')
legend('GMRES','GCRO-DR'); shg

%% GMRES-DR
disp('gmres-dr **********************************************************')
param.k = k;
param.max_it = m;
param.max_restarts = nrestarts;
param.tol = tol;
tic
for run = 1:runs
    gmresdr_matvec = 0;
    gmresdr_ip = 0;
    [x,out3] = gmres_dr(PA,Pb,param);
    gmresdr_matvec = gmresdr_matvec + out3.mv;
    gmresdr_ip = gmresdr_ip + out3.ip;
end
disp(['runtime = ' num2str(toc/runs)])
disp(['matvecs = ' num2str(gmresdr_matvec)])
disp(['ip = ' num2str(gmresdr_ip)])
disp(['trueres = ' num2str(norm(b-A*x)/norm(b))])
semilogy(0:length(out3.resvec)-1,out3.resvec,'--')
legend('GMRES','GCRO-DR','GMRES-DR'); shg

%% GMRES-SDR
disp('gmres-sdr *********************************************************')
% Note that the Signal Processing Toolbox is required for dct
opts.k = k;
param.max_it = m;
param.max_restarts = nrestarts;
param.tol = tol;
param.t = 2;       % Arnoldi truncation parameter
param.pert = 0;    % matrix A stays constant
param.verbose = 0; % no debug info computed/printed
tic
for run = 1:runs
    rng('default')    % Re-initialize for randomized sketching
    gmressdr_matvec = 0;
    gmressdr_ip = 0;
    param.U = []; param.SU = []; param.SAU = [];
    [x,out4] = gmres_sdr(PA,Pb,param);
    param.U = out4.U; param.SU = out4.SU; param.SAU = out4.SAU;
    param.hS = out4.hS;
    gmressdr_matvec = gmressdr_matvec + out4.mv;
    gmressdr_ip = gmressdr_ip + out4.ip;
end
disp(['runtime = ' num2str(toc/runs)])
disp(['matvecs = ' num2str(gmressdr_matvec)])
disp(['ip = ' num2str(gmressdr_ip)])
disp(['trueres = ' num2str(norm(b-A*x)/norm(b))])
semilogy(cumsum(out4.iters),out4.residuals,'*--')
legend('GMRES','GCRO-DR','GMRES-DR','GMRES-SDR','location','southwest');
shg

