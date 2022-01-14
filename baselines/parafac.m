% Please install Matlab code implementation of the KDD 2017 paper "SPARTan: Scalable PARAFAC2 for
% Large & Sparse Data", by Ioakeim Perros, Evangelos E. Papalexakis, Fei
% Wang, Richard Vuduc, Elizabeth Searles, Michael Thompson and Jimeng Sun
clear all;close all;

% First, specify the root directory where the Tensor Toolbox and
% N-way Toolbox folders can be found. The code has been tested using
% Tensor Toolbox v2.6 and N-Way Toolbox v3.30.
ROOTPATH = '';
addpath(strcat(ROOTPATH, 'tensor_toolbox'));
addpath(strcat(ROOTPATH, 'nway'));

% Flag indicating whether parfor capabilites will be exploited
PARFOR_FLAG = 0;
delete(gcp('nocreate'));
if (PARFOR_FLAG)
    parpool; % Set the appropriate number of workers depending on your system
else
    disp('No parallel pool enabled.');
end

K = 1000; %number of subjects (K)
J = 500; %number of variables (J)
I = 100; %max number of observations max(I_k)
sparsity = 1e-3; %sparsity of each input matrix X_k
R = 2; %target rank 

% [X, totalnnz] = create_parafac2_problem(K, J, R, sparsity, I, PARFOR_FLAG);
% fprintf(1, 'Total non-zeros: %d\n', totalnnz);

X = test();


ALG = 1;
if (ALG==1)
    disp('SPARTan execution');  % if ALG==1, then SPARTan is executed. 
else
    disp('Sparse PARAFAC2 baseline execution'); % if ALG==0, then the baseline is executed
end
Maxiters = 1000; % maximum number of iterations
Constraints = [1 1]; % non-negative constraints on both the V and S_k factors
Options = [1e-2 Maxiters 2 0 0 ALG PARFOR_FLAG]; % first argument is convergence criterion

rng('default');
[H, V, W, Q, S, FIT, IT_TIME]=parafac2_sparse_paper_version(X,R,Constraints,Options);

labels = zeros(600,1);
for i=1:600
    if S{i}(1,1) > S{i}(2,2)
        label = 0;
    else
        label = 1;
    end
    labels(i) = label;
end