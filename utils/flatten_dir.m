function flatten_dir(in_dir, out_dir, sep_symbol)
% flatten directory (1 level only)

if ~exist('sep_symbol','var'), 
  sep_symbol = '_';
end

if ~exist(out_dir,'dir'), 
  mkdir(out_dir);
end

contents = dir(in_dir); 
d_list = {contents.name};
d_list = setdiff(d_list([contents.isdir]),{'.','..'});

for d_name = d_list, 
  d_name = d_name{1};
  % d_name = fullfile(d_name, 'allFrames'); 
  contents = dir(fullfile(in_dir,d_name));
  f_list = {contents.name};
  f_list = f_list(~[contents.isdir]);
  for f_name = f_list, 
    f_name = f_name{1};
    copyfile(fullfile(in_dir,d_name,f_name), ...
      fullfile(out_dir,sprintf('%s%s%s',strrep(d_name,filesep,sep_symbol),sep_symbol,f_name)));
end


end
