%% Train YOLOX detector for SLA and BL road markings
% Uses the prepared CeyMo training/validation split without augmentation.

clear; clc; close all;

%% Prepare dataset
prepare_dataset;

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
[trainedDetector, trainInfo] = trainYOLOXObjectDetector( ...
    dsTrain, detector, options);

%% Save model
paths = project_paths();
if ~isfolder(paths.models)
    mkdir(paths.models);
end
save(fullfile(paths.models, 'trainedDetector.mat'), ...
    'trainedDetector', 'trainInfo');

fprintf('Detector saved to models/trainedDetector.mat\n');
