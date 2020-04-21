function net = fcnInitializeResNetwork4s(net, varargin)
opts.rnn = false;
opts.nh = 512;
opts.nClass = 150;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;

nh = opts.nh;
nClass = opts.nClass;

%% Remove the last layer
net.removeLayer('deconv8') ;

filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv8', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x10', 'x11', 'deconvf_3') ;

f = net.getParamIndex('deconvf_3') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;



%% build skip network

skip_inputs = {'res2ax', 'res2bx', 'res2cx'};
        
[net, classifier_out] = skipNetwork(net, skip_inputs, 256, 256, ...
    nClass, opts.newLr, 'skip2');

% Add summation layer
net.addLayer('sum4', dagnn.Sum(), ['x11', classifier_out], 'x12') ;

%% Add deconvolution layers
filters = single(bilinear_u(4, nClass, nClass)) ;
net.addLayer('deconv4', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 2, ...
  'crop', 1, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
  'x12', 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 1 ;
net.params(f).weightDecay = 1 ;
