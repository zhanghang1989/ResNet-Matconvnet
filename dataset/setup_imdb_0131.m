function imdb = setup_imdb_0131(datasetDir, varargin)

suffix = '.jpg';
allframes_subdir = 'allFrames'; 

imdb.imageDir = datasetDir;
imdb.meta.sets = {'train', 'val', 'test'}; 

seqs =dir(datasetDir);
seqs = {seqs([seqs.isdir]).name};
seqs = setdiff(seqs, {'.', '..'});

classes = dir(fullfile(datasetDir, seqs{1}));
classes = {classes([classes.isdir]).name};
imdb.meta.classes = setdiff(classes, {'.', '..', allframes_subdir});

imdb.images.name = {};
imdb.images.time = [];
imdb.images.class = [];
for i = 1:numel(seqs), 
  seqName = seqs{i};
  imdb.images.time = [imdb.images.time ; parse_seq_name(seqName,30/(24*3600),0,-1)]; % TODO fps wrong at 1st frame
  files = dir(fullfile(datasetDir, seqName, allframes_subdir, ['*' suffix]));
  fileNames = sort({files.name});
	fileNames = fileNames(2:end); % TODO fps wrong at 1st frame
  class_tbl = -1*ones(1,numel(fileNames));
  for c = 1:numel(imdb.meta.classes), 
    className = imdb.meta.classes{c}; 
    files = dir(fullfile(datasetDir, seqName, className, ['*' suffix]));
    class_tbl(ismember(fileNames,{files.name})) = c;
  end
  assert(~any(class_tbl==-1));
  imdb.images.class = [imdb.images.class class_tbl];
  imdb.images.name = [imdb.images.name ...
    cellfun(@(s) fullfile(seqName,allframes_subdir,s), fileNames,'UniformOutput',false)];
end

imdb.images.id = 1:numel(imdb.images.name);

end 

function tstr = parse_seq_name(str, off_1, off_2, off_3)

if ~exist('off_1','var') || isempty(off_1), off_1 = 0; end;
if ~exist('off_2','var') || isempty(off_2), off_2 = 0; end;
if ~exist('off_3','var') || isempty(off_3), off_3 = 0; end;

fmt_in = 'yymmddHHMMSS';
fmt_out = 'yyyy-mm-dd HH:MM:SS.FFF';

sep_locs = strfind(str,'_');
assert(numel(sep_locs)==2, 'Wrong format: %s', str);

tnum_1 = datenum(str(1:sep_locs(1)-1), fmt_in) + off_1;
tnum_2 = datenum(str(sep_locs(1)+1:sep_locs(2)-1), fmt_in) + off_2;
N = str2num(str(sep_locs(2)+1:end)) + off_3;
tnums = linspace(tnum_1, tnum_2, N); 

tstr = datestr(tnums, fmt_out);

end
