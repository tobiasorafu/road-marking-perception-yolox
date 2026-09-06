%% Analyse CeyMo Class Distribution
% Counts both image-level occurrences and object instances for the six
% road-marking classes used in this project. The official training and test
% partitions are analysed separately so the test set remains untouched.

clear; clc;

paths = project_paths();
targetClasses = {'SLA','BL','PC','JB','CL','DM'};

trainAnnDir = fullfile(paths.train, 'bbox_annotations');
testAnnDir  = fullfile(paths.test,  'bbox_annotations');

assert(isfolder(trainAnnDir), ...
    'Training annotations not found under data/ceymo/train/bbox_annotations/.');
assert(isfolder(testAnnDir), ...
    'Test annotations not found under data/ceymo/test/bbox_annotations/.');

%% Count training and test data
trainStats = countClasses(trainAnnDir, targetClasses);
testStats  = countClasses(testAnnDir, targetClasses);

%% Build summary table
className = string(targetClasses(:));
trainImages    = trainStats.imageCount(:);
testImages     = testStats.imageCount(:);
totalImages    = trainImages + testImages;
trainInstances = trainStats.instanceCount(:);
testInstances  = testStats.instanceCount(:);
totalInstances = trainInstances + testInstances;

summaryTable = table(className, trainImages, testImages, totalImages, ...
    trainInstances, testInstances, totalInstances, ...
    'VariableNames', {'Class','TrainImages','TestImages','TotalImages', ...
    'TrainInstances','TestInstances','TotalInstances'});

disp('Six-class CeyMo distribution:');
disp(summaryTable);

fprintf('\nTraining images analysed: %d\n', trainStats.numImages);
fprintf('Test images analysed:     %d\n', testStats.numImages);

%% Class imbalance relative to the smallest training class
nonzeroCounts = trainInstances(trainInstances > 0);
if ~isempty(nonzeroCounts)
    smallestCount = min(nonzeroCounts);
    imbalanceRatio = trainInstances ./ smallestCount;
    imbalanceTable = table(className, trainInstances, imbalanceRatio, ...
        'VariableNames', {'Class','TrainInstances','RelativeToSmallestClass'});

    disp('Training-set class imbalance:');
    disp(imbalanceTable);
end

%% Save the analysis for later experiment tracking
if ~isfolder(paths.results)
    mkdir(paths.results);
end

writetable(summaryTable, fullfile(paths.results, 'dataset_class_distribution.csv'));
fprintf('Saved results/dataset_class_distribution.csv\n');

%% Plot training and test instance counts
figure('Name', 'CeyMo Six-Class Distribution');
bar(categorical(className), [trainInstances testInstances]);
ylabel('Road-marking instances');
xlabel('Class');
legend('Training set', 'Test set', 'Location', 'best');
title('CeyMo Target-Class Distribution');
grid on;

%% Local function
function stats = countClasses(annotationDir, targetClasses)
xmlFiles = dir(fullfile(annotationDir, '*.xml'));
numClasses = numel(targetClasses);

stats.numImages = numel(xmlFiles);
stats.imageCount = zeros(numClasses, 1);
stats.instanceCount = zeros(numClasses, 1);

for i = 1:numel(xmlFiles)
    doc = xmlread(fullfile(annotationDir, xmlFiles(i).name));
    objects = doc.getElementsByTagName('object');
    presentInImage = false(numClasses, 1);

    for j = 0:objects.getLength()-1
        obj = objects.item(j);
        name = strtrim(char(obj.getElementsByTagName('name').item(0).getTextContent()));

        classIdx = find(strcmp(targetClasses, name), 1);
        if ~isempty(classIdx)
            stats.instanceCount(classIdx) = stats.instanceCount(classIdx) + 1;
            presentInImage(classIdx) = true;
        end
    end

    stats.imageCount = stats.imageCount + presentInImage;
end
end
