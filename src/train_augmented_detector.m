%% Train augmented YOLOX detector for SLA and BL road markings
% Applies appearance-based brightness, contrast and blur augmentation.

clear; clc; close all;

%% Prepare dataset
prepare_dataset;

%% Apply random augmentation to training data
% The validation set remains unmodified.
dsTrainAug = transform(dsTrain, @augment_data);

%% Create detector
inputSize = [224 224 3];
detector = yoloxObjectDetector("tiny-coco", ["SLA" "BL"], ...
    InputSize=inputSize);

%% Training options
options = trainingOptions("adam", ...
    InitialLearnRate=0.001, ...
    MaxEpochs=100, ...
    MiniBatchSize=70, ...
    ValidationData=dsVal, ...
    ValidationFrequency=2, ...
    Plots="training-progress", ...
    Verbose=true);

%% Train detector
[trainedDetectorAug, trainInfoAug] = trainYOLOXObjectDetector( ...
    dsTrainAug, detector, options);

%% Save model
paths = project_paths();
if ~isfolder(paths.models)
    mkdir(paths.models);
end
save(fullfile(paths.models, 'trainedDetectorAug.mat'), ...
    'trainedDetectorAug', 'trainInfoAug');

fprintf('Augmented detector saved to models/trainedDetectorAug.mat\n');
