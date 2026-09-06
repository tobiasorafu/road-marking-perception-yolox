%% Analyse CeyMo Training Set — Class Distribution
%  Counts how many images and instances of each road marking class
%  are present in the training set. Used to understand the dataset
%  before building the training pipeline.

paths = project_paths();
bboxDir = fullfile(paths.train, 'bbox_annotations');

assert(isfolder(bboxDir), ...
    'Training annotations not found. Place CeyMo training data under data/ceymo/train/.');
xmlFiles = dir(fullfile(bboxDir, '*.xml'));
numImages = length(xmlFiles);

% Initialise counters for target classes
slaImages = 0;  blImages = 0;  bothImages = 0;  neitherImages = 0;
slaInstances = 0;  blInstances = 0;

% Track all classes across the dataset
allClasses = {};

for i = 1:numImages
    xmlPath = fullfile(bboxDir, xmlFiles(i).name);
    doc = xmlread(xmlPath);
    objects = doc.getElementsByTagName('object');

    hasSLA = false;
    hasBL  = false;

    for j = 0:objects.getLength()-1
        obj = objects.item(j);
        name = char(obj.getElementsByTagName('name').item(0).getTextContent());
        allClasses{end+1} = name;

        if strcmp(name, 'SLA')
            hasSLA = true;
            slaInstances = slaInstances + 1;
        elseif strcmp(name, 'BL')
            hasBL = true;
            blInstances = blInstances + 1;
        end
    end

    if hasSLA,              slaImages = slaImages + 1;       end
    if hasBL,               blImages = blImages + 1;         end
    if hasSLA && hasBL,     bothImages = bothImages + 1;     end
    if ~hasSLA && ~hasBL,   neitherImages = neitherImages + 1; end
end

% Display target class statistics
fprintf('Total images:              %d\n', numImages);
fprintf('Images with SLA:           %d\n', slaImages);
fprintf('Images with BL:            %d\n', blImages);
fprintf('Images with both:          %d\n', bothImages);
fprintf('Images with neither:       %d\n', neitherImages);
fprintf('Total SLA instances:       %d\n', slaInstances);
fprintf('Total BL instances:        %d\n', blInstances);
fprintf('Positive images (SLA|BL):  %d\n', slaImages + blImages - bothImages);

% Display full class distribution
fprintf('\nAll classes in training set:\n');
uniqueClasses = unique(allClasses);
for k = 1:length(uniqueClasses)
    count = sum(strcmp(allClasses, uniqueClasses{k}));
    fprintf('  %-5s: %d instances\n', uniqueClasses{k}, count);
end
