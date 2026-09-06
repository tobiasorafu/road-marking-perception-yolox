%% Prepare CeyMo Dataset for MATLAB Object Detection
%  Parses Pascal VOC XML annotations for the SLA and BL classes,
%  selects a set of negative images containing visually similar
%  non-target markings, builds a ground-truth table, creates
%  training and validation datastores, and produces a visual
%  verification of the bounding box annotations.

clear; clc; close all;
rng(42);  % Fixed seed so the negative selection and split are reproducible

%% Paths
paths = project_paths();
trainDir = paths.train;
bboxDir  = fullfile(trainDir, 'bbox_annotations');
imageDir = fullfile(trainDir, 'images');

assert(isfolder(bboxDir) && isfolder(imageDir), ...
    'CeyMo training data not found. Place images and bbox_annotations under data/ceymo/train/.');

%% Parse all XML annotation files
xmlFiles = dir(fullfile(bboxDir, '*.xml'));
numFiles = length(xmlFiles);

% Preallocate a struct array to hold parsed data for every image
allData(numFiles) = struct('filename', '', 'slaBboxes', [], ...
                           'blBboxes', [], 'classes', {{}}, ...
                           'isPositive', false);

for i = 1:numFiles
    xmlPath = fullfile(bboxDir, xmlFiles(i).name);
    doc = xmlread(xmlPath);

    % Read image filename from the XML and build the full path
    imgFilename = char(doc.getElementsByTagName('filename').item(0).getTextContent());
    imgFullPath = fullfile(imageDir, imgFilename);

    objects = doc.getElementsByTagName('object');

    slaBboxes = zeros(0, 4);
    blBboxes  = zeros(0, 4);
    classesInImage = {};

    for j = 0:objects.getLength()-1
        obj = objects.item(j);
        name = char(obj.getElementsByTagName('name').item(0).getTextContent());
        classesInImage{end+1} = name;

        % Extract bounding box coordinates from the XML
        bndbox = obj.getElementsByTagName('bndbox').item(0);
        xmin = str2double(char(bndbox.getElementsByTagName('xmin').item(0).getTextContent()));
        ymin = str2double(char(bndbox.getElementsByTagName('ymin').item(0).getTextContent()));
        xmax = str2double(char(bndbox.getElementsByTagName('xmax').item(0).getTextContent()));
        ymax = str2double(char(bndbox.getElementsByTagName('ymax').item(0).getTextContent()));

        % CeyMo uses Pascal VOC format: [xmin, ymin, xmax, ymax]
        % MATLAB object detection requires: [x, y, width, height]
        x = xmin;
        y = ymin;
        w = xmax - xmin;
        h = ymax - ymin;

        if strcmp(name, 'SLA')
            slaBboxes(end+1, :) = [x, y, w, h];
        elseif strcmp(name, 'BL')
            blBboxes(end+1, :) = [x, y, w, h];
        end
    end

    allData(i).filename   = imgFullPath;
    allData(i).slaBboxes  = slaBboxes;
    allData(i).blBboxes   = blBboxes;
    allData(i).classes    = classesInImage;
    allData(i).isPositive = size(slaBboxes, 1) > 0 || size(blBboxes, 1) > 0;
end

%% Separate positive and negative images
posIdx = find([allData.isPositive]);
negIdx = find(~[allData.isPositive]);

%% Select approximately 120 negative images using class-based sampling
%  Arrow-like classes (SA, LA, SRA, RA) are overrepresented because
%  they are visually similar to SLA and more likely to cause false
%  detections. Visually distinct classes (DM, PC, JB, CL, SL) are
%  included in smaller numbers for general coverage.

targetCounts = struct('SA', 35, 'LA', 25, 'SRA', 15, 'RA', 15, ...
                      'DM', 8,  'PC', 8,  'JB',  6, 'CL',  4, 'SL', 4);

classNames = fieldnames(targetCounts);
selectedNegIdx = [];

for c = 1:length(classNames)
    cls = classNames{c};
    target = targetCounts.(cls);

    % Find negative images containing this class
    candidates = [];
    for k = 1:length(negIdx)
        if any(strcmp(allData(negIdx(k)).classes, cls))
            candidates(end+1) = negIdx(k);
        end
    end

    % Avoid selecting the same image twice
    candidates = setdiff(candidates, selectedNegIdx);

    numToSelect = min(target, length(candidates));
    if numToSelect > 0
        perm = randperm(length(candidates), numToSelect);
        selectedNegIdx = [selectedNegIdx, candidates(perm)];
    end
end

selectedNegIdx = unique(selectedNegIdx);

%% Build ground-truth table
%  Each row contains the image path and bounding boxes for SLA and BL.
%  For negative images, both columns contain empty 0x4 matrices.
%  For positive images, the non-present class also gets an empty matrix.

projectIdx = [posIdx, selectedNegIdx];
numProject = length(projectIdx);

imageFilenames = cell(numProject, 1);
slaColumn = cell(numProject, 1);
blColumn  = cell(numProject, 1);

for i = 1:numProject
    idx = projectIdx(i);
    imageFilenames{i} = allData(idx).filename;
    slaColumn{i} = allData(idx).slaBboxes;
    blColumn{i}  = allData(idx).blBboxes;
end

groundTruthTable = table(imageFilenames, slaColumn, blColumn, ...
    'VariableNames', {'imageFilename', 'SLA', 'BL'});

%% Stratified 85/15 training-validation split
%  Each image is categorised as SLA-positive, BL-positive, or negative.
%  The split is applied within each category so that both subsets
%  contain representative proportions of all three groups.

labels = zeros(numProject, 1);
for i = 1:numProject
    hasSLA = size(groundTruthTable.SLA{i}, 1) > 0;
    hasBL  = size(groundTruthTable.BL{i}, 1) > 0;
    if hasSLA
        labels(i) = 1;  % SLA
    elseif hasBL
        labels(i) = 2;  % BL
    else
        labels(i) = 3;  % Negative
    end
end

trainIdx = [];
valIdx   = [];

for cat = 1:3
    catIdx = find(labels == cat);
    catIdx = catIdx(randperm(length(catIdx)));
    nVal   = round(0.15 * length(catIdx));
    nTrain = length(catIdx) - nVal;
    trainIdx = [trainIdx; catIdx(1:nTrain)];
    valIdx   = [valIdx;   catIdx(nTrain+1:end)];
end

gtTrain = groundTruthTable(trainIdx, :);
gtVal   = groundTruthTable(valIdx, :);

%% Print dataset summary
fprintf('Project dataset:  %d images\n', numProject);
fprintf('Training set:     %d images\n', height(gtTrain));
fprintf('Validation set:   %d images\n', height(gtVal));

fprintf('\nTraining subset:\n');
fprintf('  SLA instances:   %d\n', sum(cellfun(@(x) size(x,1), gtTrain.SLA)));
fprintf('  BL instances:    %d\n', sum(cellfun(@(x) size(x,1), gtTrain.BL)));
fprintf('  Negative images: %d\n', sum(cellfun(@(x) size(x,1), gtTrain.SLA) == 0 & ...
                                       cellfun(@(x) size(x,1), gtTrain.BL) == 0));

fprintf('\nValidation subset:\n');
fprintf('  SLA instances:   %d\n', sum(cellfun(@(x) size(x,1), gtVal.SLA)));
fprintf('  BL instances:    %d\n', sum(cellfun(@(x) size(x,1), gtVal.BL)));
fprintf('  Negative images: %d\n', sum(cellfun(@(x) size(x,1), gtVal.SLA) == 0 & ...
                                       cellfun(@(x) size(x,1), gtVal.BL) == 0));

%% Create datastores
imdsTrain = imageDatastore(gtTrain.imageFilename);
bxdsTrain = boxLabelDatastore(gtTrain(:, 2:end));
dsTrain   = combine(imdsTrain, bxdsTrain);

imdsVal = imageDatastore(gtVal.imageFilename);
bxdsVal = boxLabelDatastore(gtVal(:, 2:end));
dsVal   = combine(imdsVal, bxdsVal);

%% Visual verification
%  Overlay bounding boxes on six sample images (2 SLA, 2 BL, 2 negative)
%  to confirm that the coordinate conversion and class labels are correct.

figure('Name', 'Bounding Box Verification', 'Position', [100 100 1200 800]);

verifySLA = find(cellfun(@(x) size(x,1), gtTrain.SLA) > 0, 2);
verifyBL  = find(cellfun(@(x) size(x,1), gtTrain.BL) > 0, 2);
verifyNeg = find(cellfun(@(x) size(x,1), gtTrain.SLA) == 0 & ...
                 cellfun(@(x) size(x,1), gtTrain.BL) == 0, 2);
verifyIdx = [verifySLA; verifyBL; verifyNeg];

for p = 1:min(6, length(verifyIdx))
    subplot(2, 3, p);
    idx = verifyIdx(p);
    img = imread(gtTrain.imageFilename{idx});
    imshow(img); hold on;

    % SLA boxes in green
    slaBoxes = gtTrain.SLA{idx};
    for b = 1:size(slaBoxes, 1)
        rectangle('Position', slaBoxes(b,:), 'EdgeColor', 'g', 'LineWidth', 2);
        text(slaBoxes(b,1), slaBoxes(b,2)-10, 'SLA', 'Color', 'g', ...
             'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'k');
    end

    % BL boxes in cyan
    blBoxes = gtTrain.BL{idx};
    for b = 1:size(blBoxes, 1)
        rectangle('Position', blBoxes(b,:), 'EdgeColor', 'c', 'LineWidth', 2);
        text(blBoxes(b,1), blBoxes(b,2)-10, 'BL', 'Color', 'c', ...
             'FontSize', 10, 'FontWeight', 'bold', 'BackgroundColor', 'k');
    end

    [~, fname] = fileparts(gtTrain.imageFilename{idx});
    if size(slaBoxes,1) > 0
        titleStr = sprintf('%s — SLA', fname);
    elseif size(blBoxes,1) > 0
        titleStr = sprintf('%s — BL', fname);
    else
        titleStr = sprintf('%s — Negative', fname);
    end
    title(titleStr, 'FontSize', 9);
    hold off;
end

sgtitle('Bounding Box Verification');
