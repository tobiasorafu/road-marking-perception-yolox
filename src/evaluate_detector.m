%% Evaluate trained YOLOX detector on the official CeyMo test set
% The official test set is kept separate from training and validation.
% AP is evaluated at IoU = 0.5. Detection counts use confidence = 0.2.

clear; clc; close all;

paths = project_paths();
modelPath = fullfile(paths.models, 'trainedDetector.mat');
assert(isfile(modelPath), ...
    'Trained detector not found. Run train_detector.m first.');
load(modelPath, 'trainedDetector');

%% Load test annotations
[gtTestTable, numTest] = parse_ceymo_test_set(paths.test);

slaInstances = sum(cellfun(@(x) size(x,1), gtTestTable.SLA));
blInstances  = sum(cellfun(@(x) size(x,1), gtTestTable.BL));
fprintf('Test set: %d images, %d SLA instances, %d BL instances\n', ...
    numTest, slaInstances, blInstances);

%% Run detector
allBoxes  = cell(numTest, 1);
allScores = cell(numTest, 1);
allLabels = cell(numTest, 1);

for i = 1:numTest
    img = imread(gtTestTable.imageFilename{i});
    [boxes, scores, labels] = detect(trainedDetector, img, Threshold=0.2);

    allBoxes{i} = boxes;
    allScores{i} = scores;

    if isempty(labels)
        allLabels{i} = categorical(strings(0,1), ["SLA", "BL"]);
    else
        allLabels{i} = labels;
    end

    if mod(i,100) == 0
        fprintf('Processed %d/%d images\n', i, numTest);
    end
end

%% MATLAB object-detection metrics
resultsTable = table(allBoxes, allScores, allLabels, ...
    'VariableNames', {'Boxes', 'Scores', 'Labels'});

imdsTest = imageDatastore(gtTestTable.imageFilename);
bxdsTest = boxLabelDatastore(gtTestTable(:,2:end));
dsTest = combine(imdsTest, bxdsTest);

metrics = evaluateObjectDetection(resultsTable, dsTest, 0.5, Verbose=false);
classMetrics = metrics.ClassMetrics;
classNames = string(metrics.ClassNames);

slaIdx = find(classNames == "SLA", 1);
blIdx  = find(classNames == "BL", 1);
assert(~isempty(slaIdx) && ~isempty(blIdx), 'SLA or BL metrics were not found.');

slaAP = classMetrics.APOverlapAvg(slaIdx);
blAP  = classMetrics.APOverlapAvg(blIdx);
mAP   = mean([slaAP, blAP]);

fprintf('\nAverage Precision (IoU = 0.5):\n');
fprintf('  SLA AP: %.3f\n', slaAP);
fprintf('  BL AP:  %.3f\n', blAP);
fprintf('  mAP:    %.3f\n', mAP);

%% Detection counts at the project operating threshold
confidenceThreshold = 0.2;
[slaTP, slaFP, slaFN] = count_detections( ...
    allBoxes, allScores, allLabels, gtTestTable.SLA, "SLA", confidenceThreshold);
[blTP, blFP, blFN] = count_detections( ...
    allBoxes, allScores, allLabels, gtTestTable.BL, "BL", confidenceThreshold);

slaPrecision = slaTP / max(slaTP + slaFP, 1);
blPrecision  = blTP / max(blTP + blFP, 1);

fprintf('\nDetection counts (confidence = %.1f, IoU = 0.5):\n', confidenceThreshold);
fprintf('  SLA: TP=%d, FP=%d, FN=%d, precision=%.3f\n', ...
    slaTP, slaFP, slaFN, slaPrecision);
fprintf('  BL:  TP=%d, FP=%d, FN=%d, precision=%.3f\n', ...
    blTP, blFP, blFN, blPrecision);

%% Precision-recall curves
figure('Name','Precision-Recall Curves','Position',[100 100 900 400]);

tiledlayout(1,2);
nexttile;
plot(classMetrics.Recall{slaIdx}, classMetrics.Precision{slaIdx}, ...
    'LineWidth', 2);
xlabel('Recall'); ylabel('Precision');
title(sprintf('SLA - AP %.3f', slaAP));
grid on; xlim([0 1]); ylim([0 1]);

nexttile;
plot(classMetrics.Recall{blIdx}, classMetrics.Precision{blIdx}, ...
    'LineWidth', 2);
xlabel('Recall'); ylabel('Precision');
title(sprintf('BL - AP %.3f', blAP));
grid on; xlim([0 1]); ylim([0 1]);

%% Local functions
function [gtTable, numImages] = parse_ceymo_test_set(testDir)
    bboxDir = fullfile(testDir, 'bbox_annotations');
    imageDir = fullfile(testDir, 'images');

    assert(isfolder(bboxDir) && isfolder(imageDir), ...
        'CeyMo test data not found under data/ceymo/test/.');

    xmlFiles = dir(fullfile(bboxDir, '*.xml'));
    numImages = numel(xmlFiles);

    imageFilenames = cell(numImages,1);
    slaColumn = cell(numImages,1);
    blColumn = cell(numImages,1);

    for i = 1:numImages
        doc = xmlread(fullfile(bboxDir, xmlFiles(i).name));
        filename = char(doc.getElementsByTagName('filename').item(0).getTextContent());
        imageFilenames{i} = fullfile(imageDir, filename);

        objects = doc.getElementsByTagName('object');
        slaBoxes = zeros(0,4);
        blBoxes = zeros(0,4);

        for j = 0:objects.getLength()-1
            obj = objects.item(j);
            name = char(obj.getElementsByTagName('name').item(0).getTextContent());
            if ~strcmp(name,'SLA') && ~strcmp(name,'BL')
                continue;
            end

            box = obj.getElementsByTagName('bndbox').item(0);
            xmin = str2double(char(box.getElementsByTagName('xmin').item(0).getTextContent()));
            ymin = str2double(char(box.getElementsByTagName('ymin').item(0).getTextContent()));
            xmax = str2double(char(box.getElementsByTagName('xmax').item(0).getTextContent()));
            ymax = str2double(char(box.getElementsByTagName('ymax').item(0).getTextContent()));
            matlabBox = [xmin, ymin, xmax-xmin, ymax-ymin];

            if strcmp(name,'SLA')
                slaBoxes(end+1,:) = matlabBox;
            else
                blBoxes(end+1,:) = matlabBox;
            end
        end

        slaColumn{i} = slaBoxes;
        blColumn{i} = blBoxes;
    end

    gtTable = table(imageFilenames, slaColumn, blColumn, ...
        'VariableNames', {'imageFilename','SLA','BL'});
end

function [tp, fp, fn] = count_detections(boxCells, scoreCells, labelCells, ...
        gtCells, className, confidenceThreshold)
    tp = 0; fp = 0; fn = 0;

    for i = 1:numel(boxCells)
        boxes = boxCells{i};
        scores = scoreCells{i};
        labels = labelCells{i};
        gtBoxes = gtCells{i};

        keep = scores >= confidenceThreshold & string(labels) == className;
        predBoxes = boxes(keep,:);
        matched = false(size(gtBoxes,1),1);

        for d = 1:size(predBoxes,1)
            bestIoU = 0;
            bestGt = 0;
            for g = 1:size(gtBoxes,1)
                if matched(g), continue; end
                iou = bboxOverlapRatio(predBoxes(d,:), gtBoxes(g,:));
                if iou > bestIoU
                    bestIoU = iou;
                    bestGt = g;
                end
            end

            if bestIoU >= 0.5
                tp = tp + 1;
                matched(bestGt) = true;
            else
                fp = fp + 1;
            end
        end

        fn = fn + sum(~matched);
    end
end
