function imdb = setup_imdb_minc(dataDir, varargin)
opts.seed = 1 ;
opts.joint = 0;
opts.datasetName = 'minc';
opts = vl_argparse(opts, varargin) ;

%rng(opts.seed,'twister') ;

imdb.imageDir = fullfile(dataDir, 'images') ;

cats = dir(imdb.imageDir) ;
cats = cats([cats.isdir] & ~ismember({cats.name}, {'.','..'})) ;
imdb.classes.name = {cats.name} ;
imdb.classes.description = imdb.classes.name ;
imdb.images.id = [] ;
imdb.sets = {'train', 'validate', 'test'} ;

for c=1:numel(cats)
  ims = dir(fullfile(imdb.imageDir, imdb.classes.name{c}, '*.jpg'));
  imdb.images.name{c} = cellfun(@(S) fullfile(imdb.classes.name{c}, S), ...
    {ims.name}, 'Uniform', 0);
  imdb.images.class{c} = c * ones(1,numel(ims)) ;
  if numel(ims) ~= 2500, error('MINC folder appears inconsistent') ; end
end
imdb.images.name = horzcat(imdb.images.name{:}) ;
imdb.images.class = horzcat(imdb.images.class{:}) ;
imdb.images.id = 1:numel(imdb.images.name) ;

for s = 1:3
  list = textread(fullfile(dataDir, 'labels', ...
    sprintf('%s%d.txt', imdb.sets{s}, opts.seed)),'%s') ;
  list = strrep(list, '/', '\');
  list = strrep(list, 'images\', '');
  ok = ismember(imdb.images.name, list) ;
  imdb.images.set(find(ok)) = s ;
end
if any(~ismember(imdb.images.set, 1:3)), error('MINC appears inconsistent') ; end

imdb.segments = imdb.images ;
imdb.segments.imageId = imdb.images.id ;
imdb.segments.mask = strrep(imdb.images.name, 'image', 'mask') ;

imdb.meta.classes = imdb.classes.name ;
imdb.meta.inUse = true(1, numel(imdb.meta.classes)) ;
imdb.segments.difficult = false(1, numel(imdb.segments.id)) ;
imdb.images.label = imdb.images.class ;
