function [net, info] = res_mnist(m, varargin)
% res_cifar(20, 'modelType', 'resnet', 'reLUafterSum', false,...
% 'expDir', 'data/exp/cifar-resNOrelu-20', 'gpus', [2])
%setup;
opts.modelType = 'plain' ;
opts.preActivation = false;
opts.reLUafterSum = false;
opts.shortcutBN = false;
[opts, varargin] = vl_argparse(opts, varargin) ;

if opts.preActivation ,
opts.expDir = fullfile('exp', ...
  sprintf('mnist-%s-%d', opts.modelType,m)) ;
else
     opts.expDir = fullfile('exp', ...
  sprintf('mnist-resnet-Pre-%d',m)) ;
end
opts.dataDir = fullfile('/media/wajih/Disk1 500 GB/Onus/RnD/DataSet/handwritten/','mnist') ;
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
    net = res_mnist_preactivation_init(m) ;
else
    net = res_mnist_init(m, 'networkType', opts.modelType, ...
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
  imdb = getMnistImdb(opts) ;
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
    loc(2):loc(2)+sz(2)-1,:, batch); 
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
function imdb = getMnistImdb(opts)
% -------------------------------------------------------------------------
% Preapre the imdb structure, returns image data with mean image subtracted
downloadPath = strcat(opts.dataDir,'/mnist_uint8.mat');

if ~exist(strcat(downloadPath), 'file')
  url = 'https://github.com/wajihullahbaig/DeepNeuralNetworks/tree/master/data/mnist_uint8.mat' ;
  fprintf('downloading %s\n', url) ;  
  websave(downloadPath,url)
  
end
file_set = uint8([ones(1, 6), 3]);
dataset = load(downloadPath);
data = cell(1, 7);
labels = cell(1, 7);
sets = cell(1, 7);

fullTrainingData = double(reshape(dataset.train_x',28,28,60000))/255;  
fullTrainingData = repmat(reshape(fullTrainingData,[28,28,1,60000]),1,1,1,1);
fullTrainingLabels = double(dataset.train_y) * (1:size(dataset.train_y,2)).';
batchSize = 10000;
for i= 1:6  
  data{i} = fullTrainingData(:,:,:,(i - 1) * batchSize + 1 : i * batchSize);  
  batch = fullTrainingLabels((i - 1) * batchSize + 1 : i * batchSize,1);
  labels{i} = batch'; % Index from 1
  sets{i} = repmat(file_set(i), size(labels{i}));
end

fullTestingData = double(reshape(dataset.test_x',28,28,10000))/255; 
data{7} =  repmat(reshape(fullTestingData,[28,28,1,10000]),1,1,1,1);  
labels{7} = (double(dataset.test_y) * (1:size(dataset.test_y,2)).')';
sets{7} = repmat(file_set(7), size(labels{7}));
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

% % normalize by image mean and std as suggested in `An Analysis of
% % Single-Layer Networks in Unsupervised Feature Learning` Adam
% % Coates, Honglak Lee, Andrew Y. Ng
% 
% if opts.contrastNormalization
%   z = reshape(data,[],70000) ;
%   z = bsxfun(@minus, z, mean(z,1)) ;
%   n = std(z,0,1) ;
%   z = bsxfun(@times, z, mean(n) ./ max(n, 36)) ;
%   data = reshape(z, 28, 28, 3,[]) ; % Later lets try 28,28,1 reshape
% end
% 
% if opts.whitenData
%   z = reshape(data,[],70000) ;
%   W = z(:,set == 1)*z(:,set == 1)'/70000 ;
%   [V,D] = eig(W) ;
%   % the scale is selected to approximately preserve the norm of W
%   d2 = diag(D) ;
%   en = sqrt(mean(d2)) ;
%   z = V*diag(en./max(sqrt(d2), 10))*V'*z ;
%   data = reshape(z, 28, 28,3, []) ;
% end

clNames.label_names = {'0','1','2','3','4','5','6','7','8','9'}';

imdb.images.data = data ;
imdb.images.labels = single(cat(2, labels{:})) ;
imdb.images.set = set;
imdb.meta.sets = {'train', 'val', 'test'} ;
imdb.meta.classes = clNames.label_names;
imdb.meta.dataMean = dataMean;
imdb.meta.meanType = opts.meanType; 
imdb.meta.whitenData = opts.whitenData;
imdb.meta.contrastNormalization = opts.contrastNormalization;
