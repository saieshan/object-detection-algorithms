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
% Create Faster R-CNN Detection Network
% ----------------------------------------------------------------

inputSize = [224 224 3];

preprocessedTrainingData = transform(trainingData, @(data)preprocessData(data,inputSize));
numAnchors = 3;
anchorBoxes = estimateAnchorBoxes(preprocessedTrainingData,numAnchors)

featureExtractionNetwork = resnet50;

featureLayer = 'activation_40_relu';

numClasses = width(vehicleDataset)-1;

lgraph = fasterRCNNLayers(inputSize,numClasses,anchorBoxes,featureExtractionNetwork,featureLayer);

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
% Train Faster R-CNN
% ----------------------------------------------------------------

options = trainingOptions('sgdm',...
    MaxEpochs = 10,...
    MiniBatchSize = 2,...
    InitialLearnRate = 1e-3,...
    VerboseFrequency = 1, ... 
    CheckpointPath = tempdir,...
    Shuffle = every-epoch', ...
    Plots = training-progress');
    
[detector, info] = trainFasterRCNNObjectDetector(preprocessedTrainingData,lgraph,options,'NegativeOverlapRange',[0 0.3],'PositiveOverlapRange',[0.6 1]);

counter = 0;
for j = 1:261
    I = imread(testDataTbl.imageFilename{j});
    I = imresize(I,inputSize(1:2));
    [bboxes,scores] = detect(detector,I);
    n = testDataTbl.imageFilename{j};
    y = 0;

    startPat = wildcardPattern + "/";
    endPat = ".jpg";
    newStr = extractBetween(n,startPat,endPat);
    iName = "faster_rcnn_" + newStr;

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

testData = transform(testData,@(data)preprocessData(data,inputSize));

detectionResults = detect(detector,testData,'MinibatchSize',4);

[ap, recall, precision] = evaluateDetectionPrecision(detectionResults,testData);

figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
