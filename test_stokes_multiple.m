% TEST_STOKES_MULTIPLE
% Compare GCRO-DR and GMRES-SDR on a sequence of 5 Stokes problems

clear all
close all
clc

% Download Stokes matrix from https://sparse.tamu.edu/VLSI/vas_stokes_1M
load('vas_stokes_1M.mat'); 
A = Problem.A; n = size(A,1);

% ILU preconditioner
[L,U] = ilu(A);
PA = @(x) U\(L\(A*x));

m = 100;          % max number of Arnoldi iterations
nrestarts = 10;   % max number of restarts
k = 20;           % recycling subspace dimension
tol = 1e-6;       % convergence tolerance
num_problems = 5; % number of problems

% rhs vectors with preconditioning
rng('default')
B = randn(n,num_problems);
PB = U\(L\B);

%% GCRO-DR
disp('gcrodr ************************************************************')
opts.cyclelength = m;
opts.nrestarts = nrestarts;
opts.k = k;
opts.tol = tol;
opts.isOutNMV = 1;
opts.isOutU = 1;
total_resvec = [];
opts.U = [];   % initial recycling subspace
tic
for j = 1:num_problems
    b = B(:,j);
    Pb = PB(:,j);
    % normalize Pb to compare methods using relative vs absolute residual
    bet = norm(Pb); 
    Pb = Pb/bet;
    b = b/norm(bet);
    [x,resvec,out1] = gcrodr(PA,Pb,opts);
    total_resvec = [ total_resvec , resvec ];
    opts.U = out1.U;
end
toc
semilogy(0:length(total_resvec)-1,total_resvec, '-.'); 
legend('GCRO-DR'); shg
hold on

%% GMRES-SDR
disp('gmres-sdr *********************************************************')
rng('default')
param.max_it = m;
param.max_restarts = nrestarts;
param.tol = tol;
param.k = k;
param.t = 2;       % Arnoldi truncation parameter
param.pert = 0;    % matrix A stays constant
total_resvec_gmressdr = [];
param.U = []; param.SU = []; param.SAU = []; % initial recycling subspaces
tic
for j = 1:num_problems
    b = B(:,j);
    Pb = PB(:,j);
    bet = norm(Pb);
    Pb = Pb/bet; 
    b = b/norm(bet);
    [x,out2] = gmres_sdr(PA,Pb,param);
    sres = out2.sres; sres(1) = 1;
    total_resvec_gmressdr = [ total_resvec_gmressdr, sres ];
    % now pass recycling subspaces and sketching operator to next call
    param.U = out2.U; param.SU = out2.SU; param.SAU = out2.SAU;
    param.hS = out2.hS;
end
toc
semilogy(0:length(total_resvec_gmressdr)-1,total_resvec_gmressdr)
legend('GCRO-DR','GMRES-SDR','location','northwest'); shg
title('Stokes (five systems)')
xlabel('Number of iterations');
ylabel('Relative residual norm');

