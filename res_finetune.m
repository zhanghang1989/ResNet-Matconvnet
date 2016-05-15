function [net, info] = res_finetune(varargin)
% res_finetune('datasetName', 'minc', 'datafn',...
% @setup_imdb_minc, 'nClasses', 23, 'gpus',[1 2]);
% res_finetune('gpus',[2]);

setup;

opts.datasetName = 'reflectance';
opts.datafn = @setup_imdb_reflectance;
[opts,varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data',opts.datasetName) ;
opts.imdb       = [];
opts.expDir     = fullfile('data','exp_ft', opts.datasetName) ;
opts.baseModel = 50;
opts.numEpochs  = [10 55]; 
opts.networkType = 'resnet' ;
opts.batchNormalization = true ;
opts.nClasses = 21;
opts.bn = true;
opts.whitenData = true;
opts.contrastNormalization = true;
opts.meanType = 'pixel'; % 'pixel' | 'image'
opts.border = [4 4 4 4]; % tblr
opts.gpus = []; 
opts.batchSize = 32;
opts.checkpointFn = [];
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.train.learningRate = [0.05*ones(1,10) 0.01*ones(1,10) 0.001*ones(1,10)...
    0.0001*ones(1,55)]; 
opts.train.momentum = 0.9;
opts.train.gpus = [];
opts.train = vl_argparse(opts.train, varargin) ;

if ~exist(opts.expDir, 'dir'), vl_xmkdir(opts.expDir) ; end
opts.numFetchThreads = 12 ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
if isempty(opts.imdb), 
  imdb = get_imdb(opts.datasetName, 'func', opts.datafn); 
else
  imdb = opts.imdb;
end

opts.train.train = find(imdb.images.set==1);
opts.train.val = find(imdb.images.set==3); 

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------
net = res_finetune_test(opts.baseModel, );%res_finetune_init(imdb, opts.baseModel);%


% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% Set the image average (use either an image or a color)
%net.meta.normalization.averageImage = averageImage ;
net.meta.normalization.averageImage = rgbMean ;

% Set data augmentation statistics
[v,d] = eig(rgbCovariance) ;
net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
clear v d ;

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------
func = @(a) ~isempty(a.params) ;
trainable_layers = find(arrayfun(@(l) func(l),net.layers)); 

p = arrayfun(@(l) net.layers(l) , trainable_layers); 
p = arrayfun(@(l) net.params(net.getParamIndex(l.params(1))), p, 'UniformOutput',false);
lr = cellfun(@(l) l.learningRate, p); 
layers_for_update = {trainable_layers(end), trainable_layers}; 

% tune last layer --> tune all layers
trainfn = @cnn_train_dag_check;

for s = 1:numel(opts.numEpochs), 
  if opts.numEpochs(s)<1, continue; end
  for i = 1:numel(trainable_layers), 
    l = trainable_layers(i); 
    if ismember(l,layers_for_update{s}), 
      net.layers(l).learningRate = lr(i); 
    else
      net.layers(l).learningRate = lr(i)*0; 
    end
  end
  [net, info] = trainfn(net, imdb, getBatchFn(opts, net.meta), ...
                          'expDir', opts.expDir, ...
                          opts.train, ...
                          'gpus', opts.gpus, ...
                          'batchSize',opts.batchSize,...
                          'numEpochs', sum(opts.numEpochs(1:s)),...
                          'derOutputs', {'loss', 1}, ...
                          'checkpointFn', opts.checkpointFn) ;
end


% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;

switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(bopts,x,y) ;
  case {'dagnn', 'resnet'}
    fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;
end

% -------------------------------------------------------------------------
function [im,labels] = getSimpleNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  labels = imdb.images.label(batch) ;
end

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

if ~isVal
  % training
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0) ;
else
  % validation: disable data augmentation
  im = cnn_imagenet_get_batch(images, opts, ...
                              'prefetch', nargout == 0, ...
                              'transformation', 'none') ;
end

if nargout > 0
  if useGpu
    im = gpuArray(im) ;
  end
  labels = imdb.images.label(batch) ;
  inputs = {'data', im, 'label', labels} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1: 101: end);
bs = 256 ;
opts.networkType = 'simplenn' ;
fn = getBatchFn(opts, meta) ;
avg = {}; rgbm1 = {}; rgbm2 = {};

for t=1:bs:numel(train)
  batch_time = tic ;
  batch = train(t:min(t+bs-1, numel(train))) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  temp = fn(imdb, batch) ;
  z = reshape(permute(temp,[3 1 2 4]),3,[]) ;
  n = size(z,2) ;
  avg{end+1} = mean(temp, 4) ;
  rgbm1{end+1} = sum(z,2)/n ;
  rgbm2{end+1} = z*z'/n ;
  batch_time = toc(batch_time) ;
  fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;

