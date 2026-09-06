%% Visualise previously generated video detections
% Expects detFrames to exist in the workspace. This helper only displays
% sampled detection results; it does not run video inference itself.

assert(exist('detFrames','var') == 1, ...
    'detFrames was not found in the workspace.');

detIdx = find(cellfun(@(x) ~isempty(x.bboxes), detFrames));
numDet = numel(detIdx);
assert(numDet > 0, 'No sampled detections were found in detFrames.');

figure('Name','Video Detections','Position',[100 100 1200 400]);
tiledlayout(1,numDet);

for p = 1:numDet
    nexttile;
    d = detFrames{detIdx(p)};
    img = d.image;

    for b = 1:size(d.bboxes,1)
        img = insertObjectAnnotation(img, 'rectangle', d.bboxes(b,:), ...
            sprintf('%s %.2f', string(d.labels(b)), d.scores(b)));
    end

    imshow(img);
    title(sprintf('Frame %d', d.frameNum));
end

sgtitle('Sampled Video Detections');
