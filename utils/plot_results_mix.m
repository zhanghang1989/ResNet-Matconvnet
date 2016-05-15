function plot_results_mix(expDir, datasetName, measures, savePath, varargin)
% Usage example: 
% plot_results_mix('data\exp','cifar',[],[],'plots',{'plain','resnet'})

opts.plots = {'resnet'};
opts = vl_argparse(opts, varargin);

if ~exist('datasetName', 'var') || isempty(datasetName),
    datasetName = 'cifar';
end
if ~exist('measures', 'var') || isempty(measures),
    if strcmpi(datasetName, 'cifar'), measures = {'error', 'time'};
    elseif strcmpi(datasetName, 'imagenet'), measures = {'error', 'error5'};
    else measures = {'error', 'error5'};
    end
end
if ~exist('savePath', 'var'),
    savePath = expDir;
end

if ischar(measures), measures = {measures}; end
if isempty(strfind(savePath,'.pdf')) || strfind(savePath,'.pdf')~=numel(savePath)-3,
    savePath = fullfile(savePath,[datasetName '-Gsummary.pdf']);
end

plots = opts.plots;
figure(2) ; clf ;
cmap = lines;


for k = 1:numel(measures),
    subplot(numel(measures),1,...
        k);
    hold on;
    leg = {}; Hs = []; nEpoches = 0;
    for p = plots
        p = char(p) ;
        list = dir(fullfile(expDir,sprintf('%s-%s-*',datasetName,p)));
        tokens = regexp({list.name}, sprintf('%s-%s-([\\d]+)',datasetName,p), 'tokens');
        Ns = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens);
        Ns = sort(Ns);
        for n=Ns,
            tmpDir = fullfile(expDir,sprintf('%s-%s-%d',datasetName,p,n));
            epoch = findLastCheckpoint(tmpDir);
            if epoch==0, continue; end
            load(fullfile(tmpDir,sprintf('net-epoch-%d.mat',epoch)),'stats');
            plot([stats.train.(measures{k})], ':','Color',...
                cmap(find(strcmp(p,plots))+find(Ns==n)*numel(plots),:),'LineWidth',1.5);
            Hs(end+1) = plot([stats.val.(measures{k})],...
                '-','Color',cmap(find(strcmp(p,plots))+find(Ns==n)*numel(plots),:),'LineWidth',1.5);
            leg{end+1} = sprintf('%s-%d',p,n);
            if numel(stats.train)>nEpoches, nEpoches = numel(stats.train); end
        end
    end
    xlabel('epoch') ;
    ylabel(sprintf('%s', measures{k}));
    title(measures{k}) ;
    legend(Hs,leg{:},'Location','NorthEast') ;
    %    axis square;
    if k<numel(measures) || k==1,
        ylim([0 .2]);
    end
    %xlim([0 nEpoches]);
    set(gca,'YGrid','on');
end

drawnow ;
print(savePath, '-dpdf') ;
end

function epoch = findLastCheckpoint(modelDir)
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;
end
