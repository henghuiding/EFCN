function EFCN_PascalVOC(varargin)
%FNCTRAIN Train IFCN model using MatConvNet

run ~/codes/matconvnet-1.0-beta20/matlab/vl_setupnn ;
addpath ~/codes/matconvnet-1.0-beta20/examples ;
addpath(genpath('~/codes/toolbox'));

%--------------------------------------------------------------------------
% Set the parameters
opts.modelType = 'fcn8s';
opts.rnn = false;
opts.imageSize = 512;
opts.layers = 6;
opts.kerSize = 5; % must be odd number
opts.stream = 'all';
opts.recursive = false;
opts.mode = 'val';
opts.resLayer = 101;
opts.newLr = 3;

% -------------------------------------------------------------------------
% pretrain model with COCO data
opts.coco = false;   
% Fine tune the pre-trained COCO model with Pascal data 
opts.cocoFinetune = false;  
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
% Almost fixed
opts.dataset = 'VOC';
opts.dataDir = '/home/hhding/datasets/VOC2012/';
opts.readFromDisk = true;
opts.nh = 512;
opts.rareExponnetial = 2;

if strcmp(opts.stream, 'stuff')
    opts.nClass = 36;
end
if strcmp(opts.stream, 'object')
    opts.nClass = 115;
end
if strcmp(opts.stream, 'all')
    opts.nClass = 21;
end
%--------------------------------------------------------------------------


vgg =  ~strncmpi(opts.modelType, 'res', 3);
if vgg
    opts.cocoModelPath = 'PascalVOC_512_COCO/batch-coco-ifcn8s-5x5-6layers-1e-3-higher3/net-epoch-15.mat';
    opts.sourceModelPath = '../imagenet/imagenet-vgg-verydeep-16.mat' ;
    % Path Setting
    networkFolder = sprintf('matconv21-batch-coco-i%s-%dx%d-%dlayers-1e-3-higher%d', ...'
        opts.modelType, opts.kerSize, opts.kerSize, opts.layers, opts.newLr);
else
    opts.sourceModelPath = sprintf('../imagenet/imagenet-resnet-%d-dag.mat', opts.resLayer);
    % Path Setting
    networkFolder = sprintf('batch-skip3-bn3-%s-%dx%d-%dlayers-%d-dag-1e-3-higher%d', ...'
        opts.modelType, opts.kerSize, opts.kerSize, opts.layers, opts.resLayer, ...
        opts.newLr);
end


if strcmp(opts.mode, 'val')
    opts.expDir = sprintf('PascalVOC_%d_COCO/%s', ...
        opts.imageSize, networkFolder);
else
     opts.expDir = sprintf('PascalVOC_%d_TrainVal/%s', ...
        opts.imageSize, networkFolder);
end


if opts.rnn,   opts.expDir = [opts.expDir, '-rnn'];  end

[opts, varargin] = vl_argparse(opts, varargin) ;

% experiment setup

if opts.coco
    opts.imdbPath = fullfile(opts.dataDir, 'imdb-train-val-coco-disk.mat') ;
else
    opts.imdbPath = fullfile(opts.dataDir, 'imdb-train-val-disk.mat') ;
end
    

opts.numFetchThreads = 1 ; % not used yet

% training options (SGD)
opts.train.batchSize = 10;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.gpus = [1,2];
opts.train.prefetch = true ;
opts.train.expDir = opts.expDir ;
opts.train.learningRate = 1e-3*[ones(1, 15) 0.1*ones(1,5) 0.01*ones(1,2)] ;
% opts.train.learningRate = 1e-4*getLr() ;
opts.train.numEpochs = numel(opts.train.learningRate) ;

opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
% Setup data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
else
    imdb = vocSetup('dataDir', opts.dataDir, ...
        'edition', '12', ...
        'includeTest', false, ...
        'includeSegmentation', true, ...
        'includeDetection', false) ;
    
    % add extra data
    imdb = vocSetupAdditionalSegmentations(imdb, 'dataDir', opts.dataDir) ;
    mkdir(opts.expDir) ;
    save(opts.imdbPath, '-struct', 'imdb', '-v7.3') ;
end

% Get training and test/validation subsets
train = find(imdb.images.set == 1 & imdb.images.segmentation) ;
val = find(imdb.images.set == 2  & imdb.images.segmentation) ;
test = find(imdb.images.set == 3) ;

if opts.coco
    train = train(train > 12180);
end

% -------------------------------------------------------------------------
% Setup model
% -------------------------------------------------------------------------

% Get initial model from VGG-VD-16
if vgg
    if ~opts.cocoFinetune
        net = fcnInitializeNetwork('sourceModelPath', opts.sourceModelPath,...
            'rnn', opts.rnn, 'kerSize', opts.kerSize, 'layers', opts.layers,...
            'nh', opts.nh,'nClass', opts.nClass, 'recursive', opts.recursive,...
            'newLr', opts.newLr) ;
        if any(strcmp(opts.modelType, {'fcn16s', 'fcn8s', 'fcn4s'}))
            % upgrade model to FCN16s
            net = fcnInitializeNetwork16s(net, 'rnn', false,...
                'nh', opts.nh, 'nClass', opts.nClass, 'newLr', opts.newLr) ;
        end
        if any(strcmp(opts.modelType, {'fcn8s', 'fcn4s'}))
            % upgrade model fto FCN8s
            net = fcnInitializeNetwork8s(net, 'rnn', false, ...
                'nh', opts.nh, 'nClass', opts.nClass, 'newLr', opts.newLr) ;
        end
        if strcmp(opts.modelType, 'fcn4s')
            % upgrade model fto FCN8s
            net = fcnInitializeNetwork4s(net, 'rnn', false, ...
                'nh', opts.nh, 'nClass', opts.nClass, 'newLr', opts.newLr) ;
        end
    else
        net = loadCOCOModel(opts.cocoModelPath);
    end
    
     
else
    if ~opts.cocoFinetune
        net = fcnInitializeResNetwork('sourceModelPath', opts.sourceModelPath,...
            'rnn', opts.rnn, 'kerSize', opts.kerSize, 'layers', opts.layers,...
            'nh', opts.nh,'nClass', opts.nClass, 'recursive', opts.recursive, ...
            'resLayer', opts.resLayer, 'newLr', opts.newLr);
    else
    end
    
     if any(strcmp(opts.modelType, {'res16s', 'res8s'}))
        % upgrade model to Res16s
        net = fcnInitializeResNetwork16s(net, 'rnn', false,...
            'nh', opts.nh, 'nClass', opts.nClass, 'resLayer', opts.resLayer, ...
            'newLr', opts.newLr) ;
     end
    
     if any(strcmp(opts.modelType, {'res8s'}))
        % upgrade model to Res16s
        net = fcnInitializeResNetwork8s(net, 'rnn', false,...
            'nh', opts.nh, 'nClass', opts.nClass, 'resLayer', opts.resLayer, ...
            'newLr', opts.newLr) ;
    end
end

% -------------------------------------------------------------------------
% Train
% -------------------------------------------------------------------------

% Setup data fetching options
bopts.numThreads = opts.numFetchThreads ;
bopts.labelStride = 2 ;
bopts.labelOffset = 1 ;
bopts.classWeights = getWeight(imdb.classFrequency, opts.rareExponnetial);
bopts.useGpu = numel(opts.train.gpus) > 0 ;
bopts.readFromDisk = opts.readFromDisk;
bopts.imageSize = opts.imageSize;
bopts.vgg = vgg;
bopts.stream = opts.stream;
bopts.dataset = opts.dataset;

% -------------------------------------------------------------------------
% Network setting
% -------------------------------------------------------------------------

% Launch SGD
if strcmp(opts.mode, 'val')
    trainSamples = train;
    bnSamples = val;
    bnSuffix = 'val';
elseif strcmp(opts.mode, 'test')
    trainSamples = [train val];
    bnSamples = [ train val ];
    bnSuffix = 'test';
end

fprintf('---------------------------------------------------\n')
fprintf('Please check the network setting... \n')
fprintf('Network Type: %s \n', opts.modelType);
fprintf('Context Network: Kernel Size %d x %d, %d layers \n', opts.kerSize, ...
    opts.kerSize, opts.layers);
fprintf('Batch size: %d \n', opts.train.batchSize);
fprintf('COCO Pretrain: %d \n', opts.coco);
fprintf('Image Size: %d \n', opts.imageSize);
fprintf('Starting learning rate %8.2e\n', opts.train.learningRate(1));
fprintf('New learning rates for ContextNetwork and SkipNetwork: %d \n', opts.newLr);
fprintf('Rare exponential constant %.1f\n', opts.rareExponnetial);
fprintf('In %s Mode: %d training examples, %d bn samples, with %s bn Suffix \n', ...
    opts.mode, numel(trainSamples), numel(bnSamples), bnSuffix);

fprintf('---------------------------------------------------\n')

info = fcn_train_dag(net, imdb, getBatchWrapper(bopts), opts.train, ...
    'train', trainSamples, ...
    'val', val) ;


% update BN statistics
% if ~opts.coco
    load(fullfile(opts.expDir, 'net-epoch-30.mat'), 'net');
    net = dagnn.DagNN.loadobj(net) ;
    updateBN(net, imdb, getBatchWrapper(bopts), opts.train, ...
        'train', bnSamples, 'mode', bnSuffix);
% end

% -------------------------------------------------------------------------
function fn = getBatchWrapper(opts)
% -------------------------------------------------------------------------
fn = @(imdb,batch) getBatch(imdb,batch,opts,'prefetch',nargout==0) ;

function classWeight = getWeight(h, k)
h = h(2:end);
h = h / sum(h);
classWeight = (k).^(max(0,ceil((log10(0.015 ./ h)))));
% classWeight = ones(1, numel(h));

function net = loadCOCOModel(cocoModelPath)
load(cocoModelPath, 'net') ;
net = dagnn.DagNN.loadobj(net);

% Modify the bias learning rate for all layers
for i = 1:numel(net.layers)-1
  if (isa(net.layers(i).block, 'dagnn.Conv') && net.layers(i).block.hasBias)
    filt = net.getParamIndex(net.layers(i).params{1}) ;
    bias = net.getParamIndex(net.layers(i).params{2}) ;
    net.params(filt).learningRate = 1;
    net.params(bias).learningRate = 2;
  elseif (isa(net.layers(i).block, 'dagnn.BatchNorm'))
      f = net.getParamIndex(net.layers(i).params{1});
      b = net.getParamIndex(net.layers(i).params{2});
      m = net.getParamIndex(net.layers(i).params{3});
      net.params(f).learningRate = 1;
      net.params(b).learningRate = 1;
      net.params(m).learningRate = 1;
  end
end


