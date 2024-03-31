% ----------------------------------------------------------------
% Load Dataset
% ----------------------------------------------------------------

unzip datasetImages.zip
data = load('vehicleDatasetGroundTruth.mat');
vehicleDataset = data.vehicleDataset;

rng(0)
shuffledIndices = randperm(height(vehicleDataset));
idx = floor(0.6 * height(vehicleDataset));

trainingIndex = 1:idx;
trainingDataTbl = vehicleDataset(shuffledIndices(trainingIndex),:);

testIndex = trainingIndex(end)+1 : length(shuffledIndices);
testDataTbl = vehicleDataset(shuffledIndices(testIndex),:);

trainImageDatastore = imageDatastore(trainingDataTbl{:,'imageFilename'});
trainBoxLabelDatastore = boxLabelDatastore(trainingDataTbl(:,'vehicle'));

testImageDatastore = imageDatastore(testDataTbl{:,'imageFilename'});
testBoxLabelDatastore = boxLabelDatastore(testDataTbl(:,'vehicle'));

trainingData = combine(trainImageDatastore,trainBoxLabelDatastore);
testData = combine(testImageDatastore,testBoxLabelDatastore);

data = read(trainingData);
I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% ----------------------------------------------------------------
% Create SSD Object Detection Network
% ----------------------------------------------------------------

net = resnet50();
lgraph = layerGraph(net);

inputSize = [300 300 3];

classNames = {'vehicle'};

idx = find(ismember({lgraph.Layers.Name},'activation_40_relu'));

removedLayers = {lgraph.Layers(idx+1:end).Name};
ssdLayerGraph = removeLayers(lgraph,removedLayers);

weightsInitializerValue = 'glorot';
biasInitializerValue = 'zeros';

extraLayers = [];

filterSize = 1;
numFilters = 256;
numChannels = 1024;
conv6_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv6_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu6_1 = reluLayer(Name = 'relu6_1');
extraLayers = [extraLayers; conv6_1; relu6_1];

filterSize = 3;
numFilters = 512;
numChannels = 256;
conv6_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Stride = [2, 2], ...
    Name = 'conv6_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu6_2 = reluLayer(Name = 'relu6_2');
extraLayers = [extraLayers; conv6_2; relu6_2];

filterSize = 1;
numFilters = 128;
numChannels = 512;
conv7_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv7_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu7_1 = reluLayer(Name = 'relu7_1');
extraLayers = [extraLayers; conv7_1; relu7_1];

filterSize = 3;
numFilters = 256;
numChannels = 128;
conv7_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Stride = [2, 2], ...
    Name = 'conv7_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu7_2 = reluLayer(Name = 'relu7_2');
extraLayers = [extraLayers; conv7_2; relu7_2];

filterSize = 1;
numFilters = 128;
numChannels = 256;
conv8_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv8_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu8_1 = reluLayer(Name = 'relu8_1');
extraLayers = [extraLayers; conv8_1; relu8_1];

filterSize = 3;
numFilters = 256;
numChannels = 128;
conv8_2 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Name = 'conv8_2', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu8_2 = reluLayer(Name ='relu8_2');
extraLayers = [extraLayers; conv8_2; relu8_2];

filterSize = 1;
numFilters = 128;
numChannels = 256;
conv9_1 = convolution2dLayer(filterSize, numFilters, NumChannels = numChannels, ...
    Padding = iSamePadding(filterSize), ...
    Name = 'conv9_1', ...
    WeightsInitializer = weightsInitializerValue, ...
    BiasInitializer = biasInitializerValue);
relu9_1 = reluLayer('Name', 'relu9_1');
extraLayers = [extraLayers; conv9_1; relu9_1];

if ~isempty(extraLayers)
    lastLayerName = ssdLayerGraph.Layers(end).Name;
    ssdLayerGraph = addLayers(ssdLayerGraph, extraLayers);
    ssdLayerGraph = connectLayers(ssdLayerGraph, lastLayerName, extraLayers(1).Name);
end

detNetworkSource = ["activation_22_relu", "activation_40_relu", "relu6_2", "relu7_2", "relu8_2"];

anchorBoxes = {[60,30;30,60;60,21;42,30];...
               [111,60;60,111;111,35;64,60;111,42;78,60];...
               [162,111;111,162;162,64;94,111;162,78;115,111];...
               [213,162;162,213;213,94;123,162;213,115;151,162];...
               [264,213;213,264;264,151;187,213]};

detector = ssdObjectDetector(ssdLayerGraph,classNames,anchorBoxes,DetectionNetworkSource=detNetworkSource,InputSize=inputSize,ModelName='ssdVehicle'); 

% ----------------------------------------------------------------
% Data Augmentation
% ----------------------------------------------------------------

augmentedTrainingData = transform(trainingData,@augmentData);

augmentedData = cell(4,1);
for k = 1:4
    data = read(augmentedTrainingData);
    augmentedData{k} = insertShape(data{1},rectangle = data{2});
    reset(augmentedTrainingData);
end

figure
montage(augmentedData,BorderSize = 10)

% ----------------------------------------------------------------
% Preprocess Training Data
% ----------------------------------------------------------------

preprocessedTrainingData = transform(augmentedTrainingData,@(data)preprocessData(data,inputSize));

data = read(preprocessedTrainingData);

I = data{1};
bbox = data{2};
annotatedImage = insertShape(I,'rectangle',bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

% ----------------------------------------------------------------
% Train SSD Object Detector
% ----------------------------------------------------------------

options = trainingOptions('sgdm', ...
        MiniBatchSize = 4, ....
        InitialLearnRate = 1e-3, ...
        MaxEpochs = 20, ...
        VerboseFrequency = 1, ...        
        CheckpointPath = tempdir, ...
        Shuffle = 'every-epoch', ...
        Plots='training-progress');

[detector, info] = trainSSDObjectDetector(preprocessedTrainingData,detector,options);

counter = 0;
for j = 1:261
    I = imread(imdsTest.Files{j});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);
    n = imdsTest.Files{j};
    y = 0;

    startPat = wildcardPattern + "vehicleImages\";
    endPat = ".jpg";
    newStr = extractBetween(n,startPat,endPat);
    iName = "ssd_" + newStr;

    try
       I = insertObjectAnnotation(I,'rectangle',bboxes,scores);
    catch
       fprintf('Error: %d | %s | %s\n',j,scores,n);
       counter=counter+1;
       y=1;
    end 
end
fprintf('Total Errors: %d \n',counter);

% ----------------------------------------------------------------
% Evaluate Detector Using Test Set
% ----------------------------------------------------------------

preprocessedTestData = transform(testData,@(data)preprocessData(data,inputSize));

detectionResults = detect(detector, preprocessedTestData, MiniBatchSize = 32);

[ap,recall,precision] = evaluateDetectionPrecision(detectionResults, preprocessedTestData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f',ap))
