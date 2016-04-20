function im = imresize_aug( im, opts )
% imresize augmented
% Hang Su
% 
% opts:    
%   .size       (default:: [256 256]) resizing target (NROWS,NCOLS)
%   .mode       ['crop']|'stretch'|'pad' 
%   .color      ['rgb']|'gray'
%   .pad_color  (default:: [0.4810 0.4505 0.3986]) padding color

defaultOpts.size = [256 256]; % [height width]
defaultOpts.mode = 'crop'; 
defaultOpts.color = 'rgb'; 
defaultOpts.pad_color = [0.4810 0.4505 0.3986]; 

% return default options when called w/o any arguments
if nargin==0,
    im = defaultOpts;
    return;
end

% populate the options not specified w/ default values
if ~exist('opts','var') || ~isstruct(opts), 
    opts = defaultOpts;
else
    missingFields = setdiff(fieldnames(defaultOpts),fieldnames(opts));
    for i=1:length(missingFields),
        opts.(missingFields{i}) = defaultOpts.(missingFields{i});
    end
end

if ischar(opts.pad_color), 
    switch opts.pad_color, 
        case 'white', 
            pad_color = reshape([1.0 1.0 1.0],[1 1 3]);
        case 'black', 
            pad_color = reshape([0.0 0.0 0.0],[1 1 3]);
        otherwise, 
            error('Unknown padding color: %s', opts.pad_color);
    end
else
    assert(numel(opts.pad_color)==3);
    pad_color = reshape(opts.pad_color,[1 1 3]);
end


if strcmpi(opts.color,'rgb'),   nch = 3;
elseif strcmpi(opts.color,'gray'), nch = 1;
else error('Color mode unsupported: %s',opts.color);
end
ar = opts.size(2)/opts.size(1); % aspect ratio

% convert color mode
if nch == 3 && size(im,3)==1,
    im = repmat(im,[1 1 3]);
elseif nch==1 && size(im,3)==3,
    im = rgb2gray(im);
end

% resize
sz = [size(im,1) size(im,2)];
if strcmpi(opts.mode,'crop'),         % mode 1: crop
    if sz(2)/sz(1) > ar,
        im_tmp = imresize(im,[opts.size(1) nan]);
        im = im_tmp(:,floor((size(im_tmp,2)-opts.size(2))/2)+(1:opts.size(2)),:);
    else
        im_tmp = imresize(im,[nan opts.size(2)]);
        im = im_tmp(floor((size(im_tmp,1)-opts.size(1))/2)+(1:opts.size(1)),:,:);
    end
elseif strcmpi(opts.mode,'stretch'),  % mode 2: stretch
    im = imresize(im, opts.size);
elseif strcmpi(opts.mode,'pad'),      % mode 3: pad
    im_tmp = ones(opts.size(1),opts.size(2),nch,'like',im);
    if isinteger(im), 
        assert(isa(im,'uint8'));
        im_tmp = bsxfun(@times, im_tmp, uint8(255*pad_color));
    else
        im_tmp = bsxfun(@times, im_tmp, pad_color);
    end
    if sz(2)/sz(1) > ar,
        im = imresize(im,[nan opts.size(2)]);
        im_tmp(floor((opts.size(1)-size(im,1))/2)+(1:size(im,1)),:,:) = im;
    else
        im = imresize(im,[opts.size(1) nan]);
        im_tmp(:,floor((opts.size(2)-size(im,2))/2)+(1:size(im,2)),:) = im;
    end
    im = im_tmp;
end

end
