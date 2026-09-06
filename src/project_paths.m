function paths = project_paths()
%PROJECT_PATHS Return repository-relative paths used by the project.

srcDir = fileparts(mfilename('fullpath'));
paths.root = fileparts(srcDir);
paths.data = fullfile(paths.root, 'data', 'ceymo');
paths.train = fullfile(paths.data, 'train');
paths.test = fullfile(paths.data, 'test');
paths.external = fullfile(paths.root, 'data', 'external_images');
paths.models = fullfile(paths.root, 'models');
paths.results = fullfile(paths.root, 'results');
end
