%% Augmentation function — appearance-based only

function data = augment_data(data)
img    = data{1};
bboxes = data{2};
labels = data{3};

img = im2single(img);

% Random brightness adjustment (-15% to +15%)
brightnessFactor = 1 + (rand() * 0.3 - 0.15);
img = img * brightnessFactor;

% Random contrast adjustment (0.8 to 1.2)
contrastFactor = 0.8 + rand() * 0.4;
meanVal = mean(img, 'all');
img = (img - meanVal) * contrastFactor + meanVal;

% Mild Gaussian blur with 50% probability
if rand() > 0.5
    img = imgaussfilt(img, 0.5 + rand() * 1.0);
end

% Clamp to valid range
img = im2uint8(min(max(img, 0), 1));

data{1} = img;
data{2} = bboxes;
data{3} = labels;
end
