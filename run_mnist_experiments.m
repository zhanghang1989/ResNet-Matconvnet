function run_mnist_experiments(Ns, MTs, varargin)
% Usage example:  
%  run_cifar_experiments([20 32 44 56 110 164 1001], 'resnet', 'gpus', [1]);
% Options: 
%   'expDir'['exp'], 'gpus'[[]], 'border'[[4 4 4 4]], 
%   and more defined in cnn_cifar.m

%vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc','verbose', '2') 
%setup(1,struct('enableGpu',true,'verbose','2','cudaMethod', 'nvcc','CudaRoot','/usr/local/cuda-7.5'));
%setup(1,struct('enableGpu',true,'verbose','2','cudaArch', '-gencode=arch=compute_20,code=\"sm_21,compute_20\" '));
%setup(1,struct('enableGpu',true,'verbose','2','cudaMethod','nvcc','CudaRoot','/usr/local/cuda-7.5','cudaArch', '-gencode=arch=compute_20,code=\"sm_21,compute_20\" '));

opts.expDir = fullfile('data', 'exp');
opts.gpus = [];
opts.preActivation = false;
opts = vl_argparse(opts, varargin); 

n_exp = numel(Ns); 
if ischar(MTs) || numel(MTs)==1, 
  if opts.preActivation, MTs='resnet-Pre'; end 
  if ischar(MTs), MTs = {MTs}; end; 
  MTs = repmat(MTs, [1, n_exp]); 
else
  assert(numel(MTs)==n_exp);
end

expRoot = opts.expDir; 

for i=1:n_exp, 
  opts.checkpointFn = @() plot_results(expRoot, 'mnist',[],[], 'plots', {MTs{i}});
  opts.expDir = fullfile(expRoot, ...
    sprintf('mnist-%s-%d', MTs{i}, Ns(i))); 
  [net,info] = res_mnist(Ns(i), 'modelType', MTs{i}, opts); 
  %plot_results(expRoot, 'mnist',[],[], 'plots', {MTs{i}});
end
