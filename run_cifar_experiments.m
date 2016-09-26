function run_cifar_experiments(Ns, MTs, varargin)
% Usage example:  
%  run_cifar_experiments([20 32 44 56 110 164 1001], 'resnet', 'gpus', [1]);
% Options: 
%   'expDir'['exp'], 'gpus'[[]], 'border'[[4 4 4 4]], 
%   and more defined in cnn_cifar.m

setup;

opts.expDir = fullfile('data', 'exp');
opts.gpus = [1];
opts.preActivation = false;
opts.reLUafterSum = true;
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
  opts.checkpointFn = @() plot_results(expRoot, 'cifar',[],[], 'plots', {MTs{i}});
  opts.expDir = fullfile(expRoot, ...
    sprintf('cifar-%s-%d', MTs{i}, Ns(i))); 
  [net,info] = res_cifar(Ns(i), 'modelType', MTs{i}, opts); 
end
