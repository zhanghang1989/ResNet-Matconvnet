function net = res_finetune_init(imdb, n)
net = sprintf('imagenet-resnet-%d-dag', n); 

if  ischar(net), 
  net_path = fullfile('data','models',[net '.mat']);
  if ~exist(net_path,'file'), 
    fprintf('Downloading model (%s) ...', net) ;
    vl_xmkdir(fullfile('data','models')) ;
    urlwrite(fullfile('http://www.vlfeat.org/matconvnet/models', ...
      [net '.mat']), net_path) ;
    fprintf(' done!\n');
  end
  net = dagnn.DagNN.loadobj(load(net_path));
end

net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 256 ;
net.meta.classes.name = imdb.meta.classes;
net.meta.classes.description = imdb.meta.classes;

% remove 'prob'
net.removeLayer(net.layers(end).name);

[h,w,in,out] = size(zeros(net.layers(end).block.size));
out = numel(net.meta.classes.name); 
% remove 'fc'
lName = net.layers(end).name;
net.removeLayer(net.layers(end).name);

pName = net.layers(end).name;
block = dagnn.Conv('size', [h,w,in,out], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
net.addLayer(lName, block, pName, lName, {[lName '_f'], [lName '_b']});
%net.params(net.layers(end).paramIndexes(1)).value = init_weight(opts, h, w, in, out, 'single');
%net.params(net.layers(end).paramIndexes(2)).value = zeros(out, 1, 'single');
p = net.getParamIndex(net.layers(end).params) ;
params = net.layers(end).block.initParams() ;
params = cellfun(@gather, params, 'UniformOutput', false) ;
[net.params(p).value] = deal(params{:}) ;

lName = net.layers(end).name;
net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');  
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;
net.addLayer('error5', dagnn.Loss('loss', 'topkerror', 'opts', {'topK', 5}), ...
  {'softmax','label'}, 'error5') ;

net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.transformation = 'stretch' ;

end
