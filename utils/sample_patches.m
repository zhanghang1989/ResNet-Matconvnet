function sample_patches(in_dir, out_dir, n_patches_per_image, patch_sz, scale_range, ext)
% sample_patches(in_dir, out_dir, n_patches_per_image, patch_sz, scale_range, ext)

if ~exist('n_patches_per_image', 'var') || isempty(n_patches_per_image), 
  n_patches_per_image = 1;
end

if ~exist('patch_sz', 'var') || isempty(patch_sz), 
  patch_sz = [256 256]; 
end

if ~exist('scale_range', 'var') || isempty(scale_range), 
  scale_range = [1]; 
end

if ~exist('ext', 'var') || isempty(ext), 
  ext = '.jpg';
end

if ~iscell(scale_range), scale_range = {scale_range, scale_range}; end
for i=1:2, 
  if numel(scale_range{i})==1, scale_range{i} = [scale_range{i} scale_range{i}]; end
end

vl_xmkdir(out_dir);

contents = dir(in_dir); 
im_list = {contents.name};
im_list = cellfun(@(s) fullfile(in_dir,s), im_list(~[contents.isdir]),'UniformOutput',false);
im_list = im_list(randperm(numel(im_list)));
n_digit = ceil(log10(numel(im_list)*n_patches_per_image+1));
fmt = sprintf('%%0%dd%s',n_digit,ext);

cnt = 0;
for idx = 1:numel(im_list), 
  im_path = im_list{idx}; 
  try
    im = imread(im_path);
  catch
    warning('\nUnable to read image: %s\n', im_path);
    continue;
  end
  sz0 = [size(im,1) size(im,2)];
  for i = 1:n_patches_per_image, 
    sz = max(1,round(patch_sz.*cellfun(@(t) rand()*(t(2)-t(1))+t(1),scale_range)));
    if any(sz>sz0), continue; end
    start_idx = [randi(sz0(1)-sz(1)+1) randi(sz0(2)-sz(2)+1)];
    patch = im(start_idx(1):start_idx(1)+sz(1)-1,start_idx(2):start_idx(2)+sz(2)-1,:);
    cnt = cnt + 1;
    imwrite(patch, fullfile(out_dir, sprintf(fmt,cnt)));
  end
  if mod(idx,10)==0, fprintf('.'); end;
  if mod(idx,200)==0, fprintf(' %d/%d\n',idx,numel(im_list)); end;
end

end
