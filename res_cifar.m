function [net, info] = res_cifar(m, varargin)
% res_cifar(20, 'modelType', 'resnet', 'reLUafterSum', false,...
% 'expDir', 'data/exp/cifar-resNOrelu-20', 'gpus', [2])
setup;
opts.modelType = 'resnet' ;
opts.preActivation = false;
opts.reLUafterSum = true;
opts.shortcutBN = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.preActivation ,
opts.expDir = fullfile('exp', ...
  sprintf('cifar-%s-%d', opts.modelType,m)) ;
else
     opts.expDir = fullfile('exp', ...
  sprintf('cifar-resnet-Pre-%d',m)) ;
end
opts.dataDir = fullfile('data','cifar') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.imdbPath = fullfile(opts.dataDir, 'imdb.mat');
opts.whitenData = true;
opts.contrastNormalization = true;
opts.meanType = 'image'; % 'pixel' | 'image'
opts.border = [4 4 4 4]; % tblr
opts.gpus = []; 
opts.checkpointFn = [];
opts = vl_argparse(opts, varargin) ;

if numel(opts.border)~=4, 
  assert(numel(opts.border)==1);
  opts.border = ones(1,4) * opts.border;
end

% -------------------------------------------------------------------------
%                                                    Prepare model and data
% -------------------------------------------------------------------------

if opts.preActivation ,
    net = res_cifar_preactivation_init(m) ;
else
    net = res_cifar_init(m, 'networkType', opts.modelType, ...
      'reLUafterSum', opts.reLUafterSum) ;
end

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
  if ~strcmpi(imdb.meta.meanType, opts.meanType) ...
    || xor(imdb.meta.whitenData, opts.whitenData) ...
    || xor(imdb.meta.contrastNormalization, opts.contrastNormalization);
    clear imdb;  
  end
end
if ~exist('imdb', 'var'), 
  imdb = getCifarImdb(opts) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

net.meta.classes.name = imdb.meta.classes(:)' ;
net.meta.dataMean = imdb.meta.dataMean; 
augData = zeros(size(imdb.images.data) + [sum(opts.border(1:2)) ...
  sum(opts.border(3:4)) 0 0], 'like', imdb.images.data); 
augData(opts.border(1)+1:end-opts.border(2), ...
  opts.border(3)+1:end-opts.border(4), :, :) = imdb.images.data; 
imdb.images.augData = augData; 

% -------------------------------------------------------------------------
%                                                                     Train
% -------------------------------------------------------------------------

trainfn = @cnn_train_dag_check;

rng('default');
[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  'gpus', opts.gpus, ...
  'val', find(imdb.images.set == 3), ...
  'derOutputs', {'loss', 1}, ...
  'checkpointFn', opts.checkpointFn) ;

% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
bopts = struct('numGpus', numel(opts.gpus)) ;
fn = @(x,y) getDagNNBatch(bopts,x,y) ;

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
if imdb.images.set(batch(1))==1,  % training
  sz0 = size(imdb.images.augData);
  sz = size(imdb.images.data);
  loc = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
  images = imdb.images.augData(loc(1):loc(1)+sz(1)-1, ...
    loc(2):loc(2)+sz(2)-1, :, batch); 
    if rand > 0.5, images=fliplr(images) ; end
else                              % validating / testing
  images = imdb.images.data(:,:,:,batch); 
end
labels = imdb.images.labels(1,batch) ;
if opts.numGpus > 0
  images = gpuArray(images) ;
end
inputs = {'image', images, 'label', labels} ;

% -------------------------------------------------------------------------
function imdb = getCifarImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
unpackPath = fullfile(opts.dataDir, 'cifar-10-batches-mat');
files = [arrayfun(@(n) sprintf('data_batch_%d.mat', n), 1:5, 'UniformOutput', false) ...
  {'test_batch.mat'}];
files = cellfun(@(fn) fullfile(unpackPath, fn), files, 'UniformOutput', false);
file_set = uint8([ones(1, 5), 3]);

if any(cellfun(@(fn) ~exist(fn, 'file'), files))
  url = 'http://www.cs.toronto.edu/~kriz/cifar-10-matlab.tar.gz' ;
  fprintf('downloading %s\n', url) ;
  untar(url, opts.dataDir) ;
end

data = cell(1, numel(files));
labels = cell(1, numel(files));
sets = cell(1, numel(files));
for fi = 1:numel(files)
  fd = load(files{fi}) ;
  data{fi} = permute(reshape(fd.data',32,32,3,[]),[2 1 3 4]) ;
  labels{fi} = fd.labels' + 1; % Index from 1
  sets{fi} = repmat(file_set(fi), size(labels{fi}));
end

set = cat(2, sets{:});
data = single(cat(4, data{:}));

% remove mean
dataMean = mean(data(:,:,:,set == 1), 4);
if strcmpi(opts.meanType, 'pixel'), 
  dataMean = mean(mean(dataMean, 1), 2); 
elseif ~strcmpi(opts.meanType, 'image'), 
  error('Unknown option: %s', opts.meanType); 
end
data = bsxfun(@minus, data, dataMean);

% normalize by image mean and std as suggested in `An Analysis of
% Single-Layer Networks in Unsupervised Feature Learning` Adam
% Coates, Honglak Lee, Andrew Y. Ng

if opts.contrastNormalization
  z = reshape(data,[],60000) ;
  z = bsxfun(@minus, z, mean(z,1)) ;
  n = std(z,0,1) ;
  z = bsxfun(@times, z, mean(n) ./ max(n, 40)) ;
  data = reshape(z, 32, 32, 3, []) ;
end

if opts.whitenData
  z = reshape(data,[],60000) ;
  W = z(:,set == 1)*z(:,set == 1)'/60000 ;
  [V,D] = eig(W) ;
  % the scale is selected to approximately preserve the norm of W
  d2 = diag(D) ;
  en = sqrt(mean(d2)) ;
  z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
  data = reshape(z, 32, 32, 3, []) ;
end

clNames = load(fullfile(unpackPath, 'batches.meta.mat'));

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.meta.dataMean = dataMean;
imdb.meta.meanType = opts.meanType; 
imdb.meta.whitenData = opts.whitenData;
imdb.meta.contrastNormalization = opts.contrastNormalization;
