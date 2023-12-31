function [W,R,Qtb] = qrupdate_hh(A,W,R,Qtb)
% incremental Householder QR for solving a least squares problem
%
% Example use:
%
%   rng('default')
%   A = randn(10,4); b = randn(10,1);
%   % initial QR
%   [W,R,Qtb] = qrupdate_hh(A,[],[],b);
%   ls1 = R(1:4,1:4)\Qtb(1:4);
%   % update
%   A = [ A, randn(10,3) ];
%   [W,R,Qtb] = qrupdate_hh(A,W,R,Qtb);
%   ls2 = R(1:7,1:7)\Qtb(1:7);
% 
%   % compare to MATLAB
%   [QQ,RR] = qr(A,0);
%   ls = RR\(QQ'*b);

if nargin < 4
    Qtb = NaN;
end

if isempty(A)
    W = []; R = [];
    return
end

if nargin < 2
    W = []; R = []; % initial call
end

[m,n] = size(A);
k = size(W,2); 
assert(m>n & k<n, 'A needs to be tall and size(W,2)<size(A,2)')
%Q = eye(m);
Rnew = A(:,k+1:n); % new columns in A
% first apply old Q' to new columns
for j = 1:k
    u = 2*Rnew'*W(:,j);
    Rnew = Rnew - W(:,j)*u'; %Product HR   (=Q'*A at any iteration)
end

R = [R,Rnew];
% now apply Householder to the new columns
W(m,n) = 0; % allocate (more) memory
for j = k+1:n
    x = zeros(m,1);
    x(j:m,1) = R(j:m,j);
    g = -sign(x(j))*norm(x);
    v = x; v(j) = x(j)-g;
    s = norm(v);
    if s ~= 0
        w = v/s; 
        W(:,j) = w; % store HH vector
        u = 2*(R'*w);
        R = R - w*u'; %Product HR   (=Q'*A at any iteration)
        %Q = Q - 2*Q*w*w'; %Product QR
        Qtb = Qtb - 2*w*(w'*Qtb);
    end
end
