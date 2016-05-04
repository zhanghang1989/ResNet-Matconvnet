function net = res_cifar_init(m, varargin)
switch m,
    case 20, n = 3;
    case 32, n = 5;
    case 44, n = 7;
    case 56, n = 9;
    case 110, n = 18;
end


opts.batchNormalization = true; 
opts.networkType = 'resnet'; % 'plain' | 'resnet'
opts = vl_argparse(opts, varargin); 

nClasses = 10;

net = dagnn.DagNN();

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.momentum = 0.9;
net.meta.trainOpts.batchSize = 128 ;
if opts.batchNormalization; 
  net.meta.trainOpts.learningRate = [0.1*ones(1,80) 0.01*ones(1,40) 0.001*ones(1,40)] ;
else
  net.meta.trainOpts.learningRate = [0.01*ones(1,80) 0.001*ones(1,80) 0.0001*ones(1,40)] ;
end
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% First conv layer
block = dagnn.Conv('size',  [3 3 3 16], 'hasBias', true, ...
                   'stride', 1, 'pad', [1 1 1 1]);
lName = 'conv0';
net.addLayer(lName, block, 'image', lName, {[lName '_f'], [lName '_b']});
add_layer_bn(net, 16, lName, 'bn0', 0.1); 
block = dagnn.ReLU('leak',0);
net.addLayer('relu0',  block, 'bn0', 'relu0');

info.lastNumChannel = 16;
info.lastIdx = 0;

% Three groups of layers
info = add_group(opts.networkType, net, n, info, 3, 16, 1, opts.batchNormalization);
info = add_group(opts.networkType, net, n, info, 3, 32, 2, opts.batchNormalization);
info = add_group(opts.networkType, net, n, info, 3, 64, 2, opts.batchNormalization); 

% Prediction & loss layers
block = dagnn.Pooling('poolSize', [8 8], 'method', 'avg', 'pad', 0, 'stride', 1);
net.addLayer('pool_final', block, sprintf('relu%d',info.lastIdx), 'pool_final');

block = dagnn.Conv('size', [1 1 info.lastNumChannel nClasses], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
lName = sprintf('fc%d', info.lastIdx+1);
net.addLayer(lName, block, 'pool_final', lName, {[lName '_f'], [lName '_b']});

% if opts.batchNormalization,
%   add_layer_bn(net, nClasses, lName, strrep(lName,'fc','bn'), 0.1); 
%   lName = strrep(lName, 'fc', 'bn'); 
% end

net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');  
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;

net.initParams();

end

% Add a group of layers containing 2n conv layers (w/ or w/o resnet skip connections)
function info = add_group(netType, net, n, info, w, ch, stride, bn)
if strcmpi(netType, 'plain'), 
  % the 1st layer in the group may downsample the activations by half
  add_block_conv(net, sprintf('%d', info.lastIdx+1), sprintf('relu%d', info.lastIdx), ...
    [w w info.lastNumChannel ch], stride, bn, true); 
  info.lastIdx = info.lastIdx + 1;
  info.lastNumChannel = ch;
  for i=2:2*n,
    add_block_conv(net, sprintf('%d', info.lastIdx+1), sprintf('relu%d', info.lastIdx), ...
      [w w ch ch], 1, bn, true);
    info.lastIdx = info.lastIdx + 1;
  end
elseif strcmpi(netType, 'resnet'), 
  info = add_block_res(net, info, [w w info.lastNumChannel ch], stride, bn, true); 
  for i=2:n, 
    info = add_block_res(net, info, [w w ch ch], 1, bn, false); 
  end
end
end

% Add a smallest residual unit (2 conv layers)
function info = add_block_res(net, info, f_size, stride, bn, isFirst)
lName0 = sprintf('relu%d',info.lastIdx); 
lName_tmp = lName0;
if stride > 1 || isFirst,  
  block = dagnn.Conv('size',[1 1 f_size(3) f_size(4)], 'hasBias',false,'stride',stride, ...
    'pad', 0);
  lName0 = [lName_tmp '_down2'];
  net.addLayer(lName0, block, lName_tmp, lName0, [lName0 '_f']);
  pidx = net.getParamIndex([lName0 '_f']);
  net.params(pidx).learningRate = 0;
  
  add_layer_bn(net, f_size(4), lName0, [lName_tmp '_d2bn'], 0.1); 
  lName0 = lName_tmp;
  lName_tmp = [lName_tmp '_d2bn'];
end

add_block_conv(net, sprintf('%d',info.lastIdx+1), lName0, f_size, stride, bn, true); 
info.lastIdx = info.lastIdx + 1;
info.lastNumChannel = f_size(4);
add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('relu%d',info.lastIdx), ...
  [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1, bn, false); 
info.lastNumChannel = f_size(4);
info.lastIdx = info.lastIdx + 1;
if bn, 
  lName1 = sprintf('bn%d', info.lastIdx);
else
  lName1 = sprintf('conv%d', info.lastIdx);
end

lName0 = lName_tmp;

net.addLayer(sprintf('sum%d',info.lastIdx), dagnn.Sum(), {lName0,lName1}, ...
sprintf('sum%d',info.lastIdx));

block = dagnn.ReLU('leak', 0); 
net.addLayer(sprintf('relu%d', info.lastIdx), block, sprintf('sum%d', info.lastIdx), ...
  sprintf('relu%d', info.lastIdx)); 
end

% Add a conv layer (followed by optional batch normalization & relu) 
function net = add_block_conv(net, out_suffix, in_name, f_size, stride, bn, relu)
block = dagnn.Conv('size',f_size, 'hasBias',false, 'stride', stride, ...
                   'pad',[ceil(f_size(1)/2-0.5) floor(f_size(1)/2-0.5)]);
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


