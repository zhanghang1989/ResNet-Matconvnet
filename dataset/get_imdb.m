function [ imdb ] = get_imdb( datasetName, varargin )
%GET_IMDB Get imdb structure for the specified dataset
% datasetName 
%   should be name of a directory under '/data'
% 'func'
%   the function that actually builds the imdb 
%   default: @setup_imdb_generic
% 'rebuild'
%   whether to rebuild imdb if one exists already
%   default: false


args.func = @setup_imdb_generic;
args.rebuild = false;
args.seed = 1;
if ischar(datasetName) && ~isempty(datasetName) , args.datasetName = datasetName;
else args.datasetName = 'ILSVRC2012'; end

args = vl_argparse(args,varargin);

datasetDir = fullfile('data',args.datasetName);
imdbPath = fullfile(datasetDir,'imdb.mat');

if ~exist(datasetDir,'dir'), 
    error('Unknown dataset: %s', datasetDir);
end

if exist(imdbPath,'file') && ~args.rebuild, 
    imdb = load(imdbPath);
else
    imdb = args.func(datasetDir, 'datasetName', args.datasetName, 'seed', args.seed);
    save(imdbPath,'-struct','imdb');
end

end

