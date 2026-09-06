%% Demonstrate Trained Detector on External Images
% Runs the trained YOLOX detector on images that are not part of CeyMo.
% Results are qualitative because these images have no ground-truth boxes.

clear; clc; close all;

paths = project_paths();
modelPath = fullfile(paths.models, 'trainedDetector.mat');
assert(isfile(modelPath), ...
    'Trained detector not found. Run train_detector.m first.');
load(modelPath, 'trainedDetector');

extDir = paths.external;
assert(isfolder(extDir), ...
    'External image folder not found. Place images under data/external_images/.');

imageFiles = dir(fullfile(extDir, '*.jpg'));
numImages = numel(imageFiles);
assert(numImages > 0, 'No JPG images were found in data/external_images/.');

figure('Name','External Image Demonstration','Position',[50 50 1400 800]);
tiledlayout(ceil(numImages/3), min(3,numImages));

for i = 1:numImages
    imgPath = fullfile(extDir, imageFiles(i).name);
    img = imread(imgPath);
    [bboxes, scores, labels] = detect(trainedDetector, img, Threshold=0.4);

    nexttile;
    if ~isempty(bboxes)
        annotationText = cellstr(strcat(string(labels), " ", ...
            compose('%.2f', scores)));
        img = insertObjectAnnotation(img, 'rectangle', bboxes, annotationText);
    end

    imshow(img);
    [~, filename] = fileparts(imageFiles(i).name);
    title(filename, 'Interpreter','none');
end

sgtitle('External Road Images - confidence threshold 0.4');
