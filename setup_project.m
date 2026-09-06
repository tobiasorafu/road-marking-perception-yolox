%% Set up repository paths
% Run this once after opening the repository in MATLAB.

projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'src'));

fprintf('Project source folder added to the MATLAB path.\n');
fprintf('Place the CeyMo train and test folders under data/ceymo/.\n');
