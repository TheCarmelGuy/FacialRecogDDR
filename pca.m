% --------------------------------------------------------------------
% Lehigh University - CSE
% CSE 326 - Machine Learning
% Gustavo Grinsteins
% --------------------------------------------------------------------
function [Z,vecs,vals] = pca(X,K)
  
% X is N*D input data, K is desired number of projection dimensions (assumed
% K<D).  Return values are the projected data Z, which should be N*K, vecs,
% the D*K projection matrix (the eigenvectors), and vals, which are the
% eigenvalues associated with the dimensions
  
[N D] = size(X); %rows and columns

if K > D
  error('PCA: you are trying to *increase* the dimension!');
end

% first, we have to center the data, so that the mean of each dimension
% is zero.  in other words, mean(X(:,d))==0 for all d.
mu = mean(X);

X  = X - mu;

% next, compute the covariance of the data
cov = 0;
for i = 1:1:N
    
    cov = cov + X(i,:).'*X(i,:);
    
end

C = (1./N)*cov;

% C = cov(X);

% compute the top K eigenvalues and eigenvectors of C... 
% hint: use 'eigs', or other suitable function in matlab

%all eigen vectors and values 
[V,D1] = eigs(C,K);

%pick the first k high values which are the last k values


values = diag(D1);%take the diagonal values

vecs = V;%take the vectors corresponding to
%the eigen value index

vals = diag(D1);

%vecs = V;

% project data
Z = X*vecs;

