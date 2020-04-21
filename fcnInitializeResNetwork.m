function net = fcnInitializeResNetwork(varargin)
%FCNINITIALIZEMODEL Initialize the FCN-32 model from VGG-VD-16

opts.sourceModelPath = '../imagenet/imagenet-resnet-50-dag.mat' ;
opts.rnn = false;
opts.layers = 1;
opts.kerSize = 3;
opts.nh = 512;
opts.nClass = 150;
opts.recursive = false;
opts.resLayer = 50;
opts.newLr = 1;
opts = vl_argparse(opts, varargin) ;
net = dagnn.DagNN.loadobj(load(opts.sourceModelPath)) ;

% -------------------------------------------------------------------------
%                                  Edit the model to create the FCN version
% -------------------------------------------------------------------------
% Number of classes
nClass = opts.nClass;
nh = opts.nh;

net.removeLayer('prob');
% net.removeLayer('fc365');
net.removeLayer('fc1000');
net.removeLayer('pool5');
% net.removeLayer('relu5');
% net.removeLayer('bn5');

%% adapat the network

net.addLayer('adaptation', ...
     dagnn.Conv('size', [1 1 2048 nh], 'pad', 0), ...
     'res5cx', 'res6x', {'adaptation_f','adaptation_b'});

f = net.getParamIndex('adaptation_f') ;
net.params(f).value = 1e-2*randn(1, 1, 2048, nh, 'single') ;
net.params(f).learningRate = 1 * opts.newLr;
net.params(f).weightDecay = 1 ;

f = net.getParamIndex('adaptation_b') ;
net.params(f).value = zeros(1, 1, nh, 'single') ;
net.params(f).learningRate = 2 * opts.newLr ;
net.params(f).weightDecay = 1 ;

net.addLayer('adapation_relu', ...
        dagnn.ReLU(),...
        'res6x', 'res6x1');


%% build context network
[net, ~, cn_classifier_out] = contextNetwork(net, 'res6x1', opts.kerSize,...
    nh, nh, nClass, opts.layers, opts.newLr, 'conv5', opts.recursive);

%% build skip network
% skip_inputs = {};
skip_inputs = {'res5ax', 'res5bx', 'res5cx'};
[net, skip_classifier_out] = skipNetwork(net, skip_inputs, 2048, nh, ...
    nClass, opts.newLr, 'skip5');

%%
% -------------------------------------------------------------------------
%  Summing layer
% -------------------------------------------------------------------------
if numel(cn_classifier_out) > 0
    net.addLayer('sum_1_1', dagnn.Sum(), [cn_classifier_out, skip_classifier_out], 'sum_1_out') ;
    deconv_in = 'sum_1_out';
else
    error('The depth of context network must be deeper than 1.');
end

% -------------------------------------------------------------------------
% Upsampling and prediction layer
% -------------------------------------------------------------------------


filters = single(bilinear_u(32, nClass, nClass)) ;
net.addLayer('deconv32', ...
  dagnn.ConvTranspose(...
  'size', size(filters), ...
  'upsample', 16, ...
  'crop', 8, ...
  'numGroups', nClass, ...
  'hasBias', false), ...
   deconv_in, 'prediction', 'deconvf') ;

f = net.getParamIndex('deconvf') ;
net.params(f).value = filters ;
net.params(f).learningRate = 0 ;
net.params(f).weightDecay = 1 ;

% Make the output of the bilinear interpolator is not discared for
% visualization purposes
net.vars(net.getVarIndex('prediction')).precious = 1 ;

% -------------------------------------------------------------------------
% Losses and statistics
% -------------------------------------------------------------------------

% Add loss layer
net.addLayer('objective', ...
  WeightSegmentationLoss('loss', 'idfsoftmaxlog'), ...
  {'prediction', 'label', 'classWeight'}, 'objective') ;

% Add accuracy layer
net.addLayer('accuracy', ...
  SegmentationAccuracy(), ...
  {'prediction', 'label'}, 'accuracy') ;

if 0
  figure(100) ; clf ;
  n = numel(net.vars) ;
  for i=1:n
    vl_tightsubplot(n,i) ;
    showRF(net, 'input', net.vars(i).name) ;
    title(sprintf('%s', net.vars(i).name)) ;
    drawnow ;
  end
end




