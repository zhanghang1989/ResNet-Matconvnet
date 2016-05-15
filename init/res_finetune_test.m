function net = res_finetune_test(n, varargin)
net = sprintf('imagenet-resnet-%d-dag', n); 
net_path = fullfile('data','models',[net '.mat']);
netp = dagnn.DagNN.loadobj(load(net_path));

opts.batchNormalization = true; 
opts.networkType = 'resnet'; % 'plain' | 'resnet'
opts.bottleneck = false; % only used when n is an array
opts.nClasses = 21;
nClasses = opts.nClasses;
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN();

% n -> specific configuration
if numel(n)==4, 
  Ns = n;
else
  switch n, 
    case 50, Ns = [3 4 6 3]; opts.bottleneck = true;
    case 101, Ns = [3 4 23 3]; opts.bottleneck = true; 
    case 152, Ns = [3 8 36 3]; opts.bottleneck = true; 
    otherwise, error('No configuration found for n=%d', n); 
  end 
end
if strcmpi(opts.networkType, 'plain') && opts.bottleneck, 
  error('plain network cannot be built with bottleneck layers');
end

% Meta parameters
net.meta.inputSize = [224 224 3] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 256 ;
if opts.batchNormalization; 
  net.meta.trainOpts.learningRate = [0.1*ones(1,30) 0.01*ones(1,30) 0.001*ones(1,50)] ;
else
  net.meta.trainOpts.learningRate = [0.01*ones(1,45) 0.001*ones(1,45) 0.0001*ones(1,75)] ;
end
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% First conv layer
switch n
    case 50
        block = dagnn.Conv('size',  [7 7 3 64], 'hasBias', true, ...
            'stride', 2, 'pad', [3 3 3 3]);
        lName = 'conv0';
net.addLayer(lName, block, 'data', lName, {[lName '_f'], [lName '_b']});
    case {101, 152}
        block = dagnn.Conv('size',  [7 7 3 64], 'hasBias', false, ...
            'stride', 2, 'pad', [3 3 3 3]);
        lName = 'conv0';
    net.addLayer(lName, block, 'data', lName, {[lName '_f']});
end

add_layer_bn(net, 64, lName, 'bn0', 0.1); 
block = dagnn.ReLU('leak',0);
net.addLayer('relu0',  block, 'bn0', 'relu0');

%add_block_conv(net, '0', 'image', [7 7 3 64], 2, opts.batchNormalization, true); 
block = dagnn.Pooling('poolSize', [3 3], 'method', 'max', 'pad', [0 1 0 1], 'stride', 2); 
net.addLayer('pool0', block, 'relu0', 'pool0'); 

info.lastNumChannel = 64;
info.lastIdx = 0;
info.lastName = 'pool0'; 

% Four groups of layers
info = add_group(opts.networkType, net, Ns(1), info, 3, 64,  1, opts.bottleneck, opts.batchNormalization);
info = add_group(opts.networkType, net, Ns(2), info, 3, 128, 2, opts.bottleneck, opts.batchNormalization);
info = add_group(opts.networkType, net, Ns(3), info, 3, 256, 2, opts.bottleneck, opts.batchNormalization); 
info = add_group(opts.networkType, net, Ns(4), info, 3, 512, 2, opts.bottleneck, opts.batchNormalization); 

% Prediction & loss layers
block = dagnn.Pooling('poolSize', [7 7], 'method', 'avg', 'pad', 0, 'stride', 1);
net.addLayer('pool_final', block, sprintf('relu%d',info.lastIdx), 'pool_final');

block = dagnn.Conv('size', [1 1 info.lastNumChannel nClasses], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
lName = sprintf('fc%d', info.lastIdx+1);
net.addLayer(lName, block, 'pool_final', lName, {[lName '_f'], [lName '_b']});


net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');  
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;

net.initParams();

net.meta.normalization.imageSize = net.meta.inputSize;
net.meta.inputSize = net.meta.normalization.imageSize ;
net.meta.normalization.border = 256 - net.meta.inputSize(1:2) ;
net.meta.normalization.interpolation = 'bicubic' ;
net.meta.normalization.averageImage = [] ;
net.meta.normalization.keepAspect = true ;
net.meta.augmentation.rgbVariance = zeros(0,3) ;
net.meta.augmentation.transformation = 'stretch' ;

for id=1:size(netp.params(:))
    if size(netp.params(id).value(:)) == size(net.params(id).value(:))
        net.params(id).value(:) = netp.params(id).value(:);
    else
        fprintf('skipping params %d\n', id)
    end
end

end

% Add a group of layers containing 2n/3n conv layers
function info = add_group(netType, net, n, info, w, ch, stride, bottleneck, bn)
if strcmpi(netType, 'plain'), 
   % TODO: update 'plain', add bn
  if isfield(info, 'lastName'), 
    lName = info.lastName; 
    info = rmfield(info, 'lastName');
  else
    lName = sprintf('relu%d', info.lastIdx);
  end
  % the 1st layer in the group may downsample the activations by half
  add_block_conv(net, sprintf('%d', info.lastIdx+1), lName, ...
    [w w info.lastNumChannel ch], stride, bn, true); 
  info.lastIdx = info.lastIdx + 1;
  info.lastNumChannel = ch;
  for i=2:2*n,
    add_block_conv(net, sprintf('%d', info.lastIdx+1), sprintf('relu%d', info.lastIdx), ...
      [w w ch ch], 1, bn, true);
    info.lastIdx = info.lastIdx + 1;
  end
elseif strcmpi(netType, 'resnet'), 
  info = add_block_res(net, info, [w w info.lastNumChannel ch], stride, bottleneck, bn, 1); 
  for i=2:n, 
    info = add_block_res(net, info, [w w 4*ch ch], 1, bottleneck, bn, 0); 
  end
end
end

% Add a smallest residual unit (2/3 conv layers)
function info = add_block_res(net, info, f_size, stride, bottleneck, bn, isFirst)
if isfield(info, 'lastName'), 
  lName0 = info.lastName;
  info = rmfield(info, 'lastName'); 
else
  lName0 = sprintf('relu%d',info.lastIdx); 
end
lName01 = lName0;

if stride > 1 || isFirst, 
  block = dagnn.Conv('size',[1 1 f_size(3) 4*f_size(4)], 'hasBias',false,'stride',stride, ...
    'pad', 0);
  lName_tmp = lName0;
  lName0 = [lName_tmp '_down2'];
  net.addLayer(lName0, block, lName_tmp, lName0, [lName0 '_f']);
  
  pidx = net.getParamIndex([lName0 '_f']);
  net.params(pidx).learningRate = 0;
  
  add_layer_bn(net, 4*f_size(4), lName0, [lName01 '_d2bn'], 0.1); 
  lName0 = [lName01 '_d2bn'];
end

if bottleneck, 
  add_block_conv(net, sprintf('%d',info.lastIdx+1), lName01, [1 1 f_size(3) f_size(4)], stride, bn, true); 
  info.lastIdx = info.lastIdx + 1;
  info.lastNumChannel = f_size(4);
  add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('relu%d',info.lastIdx), ...
    [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1, bn, true); 
  info.lastIdx = info.lastIdx + 1;
  add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('relu%d',info.lastIdx), ...
    [1 1 info.lastNumChannel info.lastNumChannel*4], 1, bn, false); 
  info.lastIdx = info.lastIdx + 1;
  info.lastNumChannel = info.lastNumChannel*4; 
else
  add_block_conv(net, sprintf('%d',info.lastIdx+1), lName01, f_size, stride, bn, true); 
  info.lastIdx = info.lastIdx + 1;
  info.lastNumChannel = f_size(4);
  add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('relu%d',info.lastIdx), ...
    [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1, bn, false); 
  info.lastIdx = info.lastIdx + 1;
end
if bn, 
  lName1 = sprintf('bn%d', info.lastIdx);
else
  lName1 = sprintf('conv%d', info.lastIdx);
end


% ToDo: update sum layer
if 1% f_size(3)==info.lastNumChannel, 
  net.addLayer(sprintf('sum%d',info.lastIdx), dagnn.Sum(), {lName0,lName1}, ...
    sprintf('sum%d',info.lastIdx));
else
  net.addLayer(sprintf('sum%d',info.lastIdx), dagnn.PadSum(), {lName0,lName1}, ...
    sprintf('sum%d',info.lastIdx));
end

% relu
block = dagnn.ReLU('leak', 0); 
net.addLayer(sprintf('relu%d', info.lastIdx), block, sprintf('sum%d', info.lastIdx), ...
  sprintf('relu%d', info.lastIdx)); 
end

% Add a conv layer (followed by optional batch normalization & relu) 
function net = add_block_conv(net, out_suffix, in_name, f_size, stride, bn, relu)
block = dagnn.Conv('size',f_size, 'hasBias',false, 'stride', stride, ...
                   'pad',[ceil(f_size(1)/2-0.5) floor(f_size(1)/2-0.5) ...
                   ]);
lName = ['conv' out_suffix];
net.addLayer(lName, block, in_name, lName, {[lName '_f']});

if bn, 
  add_layer_bn(net, f_size(4), lName, strrep(lName,'conv','bn'), 0.1); 
  lName = strrep(lName, 'conv', 'bn');
end
if relu, 
  block = dagnn.ReLU('leak',0);
  net.addLayer(['relu' out_suffix], block, lName, ['relu' out_suffix]);
end
end

% Add a batch normalization layer
function net = add_layer_bn(net, n_ch, in_name, out_name, lr)
block = dagnn.BatchNorm('numChannels', n_ch);
net.addLayer(out_name, block, in_name, out_name, ...
  {[out_name '_g'], [out_name '_b'], [out_name '_m']});
pidx = net.getParamIndex({[out_name '_g'], [out_name '_b'], [out_name '_m']});
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(2)).weightDecay = 0; 
net.params(pidx(3)).learningRate = lr;
net.params(pidx(3)).trainMethod = 'average'; 
end

