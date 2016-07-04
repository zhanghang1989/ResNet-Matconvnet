function net = res_cifar_preactivation_init(m)
switch m,
    case 20, n = 3; opts.bottleneck = false;
    case 32, n = 5; opts.bottleneck = false;
    case 44, n = 7; opts.bottleneck = false;
    case 56, n = 9; opts.bottleneck = false;
    case 110, n = 18; opts.bottleneck = false;
    case 164,  n = 18; opts.bottleneck = true;
    case 1001,  n = 111; opts.bottleneck = true;
    otherwise, error('No configuration found for n=%d', n);
end



nClasses = 10;
net = dagnn.DagNN();

% Meta parameters
net.meta.inputSize = [32 32 3] ;
net.meta.trainOpts.weightDecay = 0.0001 ;
net.meta.trainOpts.momentum = 0.9;
if m > 200 ,
    net.meta.trainOpts.batchSize = 64 ;
else
    net.meta.trainOpts.batchSize = 128 ;
end

net.meta.trainOpts.learningRate = [0.01*ones(1,80) 0.001*ones(1,80) 0.0001*ones(1,40)] ;
net.meta.trainOpts.numEpochs = numel(net.meta.trainOpts.learningRate) ;

% First conv layer 
block = dagnn.Conv('size',  [3 3 3 16], 'hasBias', true, ...
    'stride', 1, 'pad', [1 1 1 1]);
lName = 'conv0';
net.addLayer(lName, block, 'image', lName, {[lName '_f'], [lName '_b']});

info.lastNumChannel = 16;
info.lastIdx = 0;

% Three groups of layers
info = add_group(net, n, info, 3, 16, 1, opts);
info = add_group(net, n, info, 3, 32, 2, opts);
info = add_group(net, n, info, 3, 64, 2, opts);

% Prediction & loss layers
add_layer_bn(net, 4*64, sprintf('sum%d',info.lastIdx), 'bn_final', 0.1);
block = dagnn.ReLU('leak',0);
net.addLayer('relu_final',  block, 'bn_final', 'relu_final');
block = dagnn.Pooling('poolSize', [8 8], 'method', 'avg', 'pad', 0, 'stride', 1);
net.addLayer('pool_final', block, 'relu_final', 'pool_final');


block = dagnn.Conv('size', [1 1 info.lastNumChannel nClasses], 'hasBias', true, ...
    'stride', 1, 'pad', 0);
lName = sprintf('fc%d', info.lastIdx+1);
net.addLayer(lName, block, 'pool_final', lName, {[lName '_f'], [lName '_b']});


net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');
net.addLayer('loss', dagnn.Loss('loss', 'log'), {'softmax', 'label'}, 'loss');
net.addLayer('error', dagnn.Loss('loss', 'classerror'), {'softmax','label'}, 'error') ;

net.initParams();



% Add a group of layers containing 2n/3n conv layers
function info = add_group( net, n, info, w, ch, stride, opts)

info = add_block_res(net, info, [w w info.lastNumChannel ch], stride, true, opts);
for i=2:n,
    if opts.bottleneck,
        info = add_block_res(net, info, [w w 4*ch ch], 1, false, opts);
    else
        info = add_block_res(net, info, [w w ch ch], 1, false, opts);
    end
end



% Add a smallest residual unit (2/3 conv layers)
function info = add_block_res(net, info, f_size, stride, isFirst, opts)
if isfield(info, 'lastName'),
    lName0 = info.lastName;
    info = rmfield(info, 'lastName');
elseif info.lastIdx == 0,
    lName0 = sprintf('conv0');
else
    lName0 = sprintf('sum%d',info.lastIdx);
end

lName01 = lName0;
if isFirst,
    if opts.bottleneck,
        ch = 4*f_size(4);
    else
        ch = f_size(4);
    end
    % bn & relu
    add_layer_bn(net, f_size(3), lName0, [lName0 '_bn'], 0.1);
    block = dagnn.ReLU('leak',0);
    net.addLayer([lName0 '_relu'],  block, [lName0 '_bn'], [lName0 '_relu']);
    lName0 = [lName0 '_relu'];

    % change featuremap size and chanels
    block = dagnn.Conv('size',[1 1 f_size(3) ch], 'hasBias', false,'stride',stride, ...
        'pad', 0);
    lName_tmp = lName0;
    lName0 = [lName_tmp '_down2'];
    net.addLayer(lName0, block, lName_tmp, lName0, [lName0 '_f']);
    
    pidx = net.getParamIndex([lName0 '_f']);
    net.params(pidx).learningRate = 0;
end

if opts.bottleneck,
    add_block_conv(net, sprintf('%d',info.lastIdx+1), lName01, [1 1 f_size(3) f_size(4)], stride);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = f_size(4);
    add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('conv%d',info.lastIdx), ...
        [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1);
    info.lastIdx = info.lastIdx + 1;
    add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('conv%d',info.lastIdx), ...
        [1 1 info.lastNumChannel info.lastNumChannel*4], 1);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = info.lastNumChannel*4;
else
    add_block_conv(net, sprintf('%d',info.lastIdx+1), lName01, f_size, stride);
    info.lastIdx = info.lastIdx + 1;
    info.lastNumChannel = f_size(4);
    add_block_conv(net, sprintf('%d',info.lastIdx+1), sprintf('conv%d',info.lastIdx), ...
        [f_size(1) f_size(2) info.lastNumChannel info.lastNumChannel], 1);
    info.lastIdx = info.lastIdx + 1;
end

lName1 = sprintf('conv%d', info.lastIdx);

net.addLayer(sprintf('sum%d',info.lastIdx), dagnn.Sum(), {lName0,lName1}, ...
    sprintf('sum%d',info.lastIdx));



% Add a conv layer (followed by optional batch normalization & relu)
function net = add_block_conv(net, out_suffix, in_name, f_size, stride)

lName = ['bn' out_suffix];
add_layer_bn(net, f_size(3), in_name, lName, 0.1);

block = dagnn.ReLU('leak',0);
net.addLayer(['relu' out_suffix], block, lName, ['relu' out_suffix]);

block = dagnn.Conv('size',f_size, 'hasBias',false, 'stride', stride, ...
    'pad',[ceil(f_size(1)/2-0.5) floor(f_size(1)/2-0.5) ...
    ]);
lName = ['conv' out_suffix];
net.addLayer(lName, block, ['relu' out_suffix], lName, {[lName '_f']});



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


