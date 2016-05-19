function [net, info] = res_imagenet(n, varargin)
% Usage example on ImageNet: res_imagenet(18,'gpus', [1 2])
% Usage example for your own dataset:
% res_imagenet(18, 'datasetName', 'reflectance', 'datafn',...
%   @setup_imdb_reflectance, 'nClasses', 21, 'gpus', [1 2])

setup;

opts.datasetName = 'ILSVRC2012';
opts.datafn = @setup_imdb_imagenet;
[opts,varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile('data',opts.datasetName) ;
opts.imdb       = [];
opts.networkType = 'resnet' ;
opts.expDir     = fullfile('data','exp', ...
    sprintf('%s-%s-%d', opts.datasetName, opts.networkType , n)) ;

opts.batchNormalization = true ;
opts.nClasses = 1000;
opts.batchSize = 256;
opts.numAugments = 1 ;
opts.numEpochs = 110;
opts.bn = true;
opts.whitenData = true;
opts.contrastNormalization = true;
opts.meanType = 'image'; % 'pixel' | 'image'
opts.gpus = []; 
opts.checkpointFn = [];
[opts, varargin] = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), vl_xmkdir(opts.expDir) ; end
opts.numFetchThreads = 12 ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = res_imagenet_init(n, 'nClasses', opts.nClasses,...
                        'batchNormalization', opts.batchNormalization, ...
                        'networkType', opts.networkType) ;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------
if isempty(opts.imdb), 
  imdb = get_imdb(opts.datasetName, 'func', opts.datafn); 
else
  imdb = opts.imdb;
end

% Set the class names in the network
net.meta.classes.name = imdb.classes.name ;
net.meta.classes.description = imdb.classes.description ;

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
trainfn = @cnn_train_dag_check;

[net, info] = trainfn(net, imdb, getBatchFn(opts, net.meta), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  'gpus', opts.gpus, ...
  'batchSize',opts.batchSize,...
  'numEpochs',opts.numEpochs,...
  'val', find(imdb.images.set == 3), ...
  'derOutputs', {'loss', 1}, ...
  'checkpointFn', opts.checkpointFn) ;


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
bopts.numAugments = opts.numAugments ; 

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

