function [Q1,R1] = qrupdate_gs(A,Q,R)
% modified Gram-Schmidt update of QR factorization
% Example use:
% 
%  A = randn(10,6);
%  [Q,R] = qr(A(:,1:4),0); % partial QR
%  [Q1,R1] = qrupdate_gs(A,Q,R);
[m,n] = size(A);
k = size(Q,2);
assert(m>n & k<n, 'A needs to be tall and size(Q,2)<size(A,2)')
Q1 = zeros(m,n);
R1 = zeros(n,n);
Q1(1:m,1:k) = Q;
R1(1:k,1:k) = R;
for j = k+1:n
    w = A(:,j);
    for reo = 0:1
        for i = 1:j-1
            r = Q1(:,i)'*w;
            R1(i,j) = R1(i,j) + r;
            w = w - Q1(:,i)*r;
        end
    end
    R1(j,j) = norm(w);
    Q1(:,j) = w/R1(j,j);
end