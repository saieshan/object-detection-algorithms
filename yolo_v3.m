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
% Create YOLO v3 Object Detector
% ----------------------------------------------------------------

networkInputSize = [227 227 3];

rng(0)
trainingDataForEstimation = transform(trainingData, @(data)preprocessData(data, networkInputSize));
numAnchors = 6;
[anchors, meanIoU] = estimateAnchorBoxes(trainingDataForEstimation, numAnchors)

area = anchors(:, 1).*anchors(:, 2);
[~, idx] = sort(area, 'descend');
anchors = anchors(idx, :);
anchorBoxes = {anchors(1:3,:)
    anchors(4:6,:)
    };

baseNetwork = squeezenet;
classNames = trainingDataTbl.Properties.VariableNames(2:end);

yolov3Detector = yolov3ObjectDetector(baseNetwork, classNames, anchorBoxes, 'DetectionNetworkSource', {'fire9-concat', 'fire5-concat'}, InputSize = networkInputSize);

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

preprocessedTrainingData = transform(augmentedTrainingData, @(data)preprocess(yolov3Detector, data));

data = read(preprocessedTrainingData);

I = data{1,1};
bbox = data{1,2};
annotatedImage = insertShape(I, 'Rectangle', bbox);
annotatedImage = imresize(annotatedImage,2);
figure
imshow(annotatedImage)

reset(preprocessedTrainingData);

% ----------------------------------------------------------------
% Train Model
% ----------------------------------------------------------------

options = trainingOptions('sgdm', ...
        MiniBatchSize = 8, ....
        InitialLearnRate = 1e-3, ...
        MaxEpochs = 80, ...
        VerboseFrequency = 1, ...        
        CheckpointPath = tempdir, ...
        Shuffle = 'every-epoch', ...
        Plots='training-progress');
		
for epoch = 1:numEpochs

	[gradients, state, lossInfo] = dlfeval(@modelGradients, yolov3Detector, XTrain, YTrain, penaltyThreshold);
	gradients = dlupdate(@(g,w) g + l2Regularization*w, gradients, yolov3Detector.Learnables);
	currentLR = piecewiseLearningRateWithWarmup(iteration, epoch, learningRate, warmupPeriod, numEpochs);
	
	[yolov3Detector.Learnables, velocity] = sgdmupdate(yolov3Detector.Learnables, gradients, velocity, currentLR);
end

counter = 0;
for j = 1:261
    I = imread(imdsTest.Files{j});
    [bboxes,scores,labels] = detect(yolov3Detector,I);
    n = imdsTest.Files{j};
    y = 0;

    startPat = wildcardPattern + "vehicleImages\";
    endPat = ".jpg";
    newStr = extractBetween(n,startPat,endPat);
    iName = "yolo_v3_" + newStr;

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
% Evaluate Model
% ----------------------------------------------------------------

results = detect(yolov3Detector,testData,'MiniBatchSize',8);

% Evaluate the object detector using Average Precision metric.
[ap,recall,precision] = evaluateDetectionPrecision(results,testData);

% Plot precision-recall curve.
figure
plot(recall,precision)
xlabel('Recall')
ylabel('Precision')
grid on
title(sprintf('Average Precision = %.2f', ap))
