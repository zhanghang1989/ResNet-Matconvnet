# ResNet-Matconvnet

This repository is a Matconvnet re-implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385). You can train Deep Residual Network on ImageNet from Scratch or fine-tune pre-trained model on your own dataset. This repo is created by [Hang Zhang](http://www.hangzh.com).

### Table of Contents
0. [Get Started](#get-started)
0. [Train from Scratch](#train-from-scratch)
0. [Fine-tune Your Own](#fine-tune-your-own)
0. [Changes](#changes)

### Get Started

The code relies on [vlfeat](http://www.vlfeat.org/), and [matconvnet](http://www.vlfeat.org/matconvnet/), which should be downloaded and built before running the experiments. You can use the following commend to download them.
```sh
git clone -b v1.0 --recurse-submodules https://github.com/zhanghang1989/ResNet-Matconvnet.git
```
If you have problem with compiling, please refer to the [link](http://zhanghang1989.github.io/ResNet/#compiling-vlfeat-and-matconvnet).	
### Train from Scratch
0. **Cifar.** Reproducing Figure 6 from the original paper.
	```matlab
	run_cifar_experiments([20 32 44 56 110], 'plain', 'gpus', [1]);
	run_cifar_experiments([20 32 44 56 110], 'resnet', 'gpus', [1]);
	```
	
	<p style="text-align:center; font-size:75%; font-style: italic;">Cifar Experiments</p>

	<div style="text-align:center"><img src ="https://raw.githubusercontent.com/zhanghang1989/ResNet-Matconvnet/master/figure/plain_cifar.png" width="420" /><img src ="https://raw.githubusercontent.com/zhanghang1989/ResNet-Matconvnet/master/figure/resnet_cifar.png" width="420" /></div>	
	
	Reproducing the experiments in Facebook [blog](http://torch.ch/blog/2016/02/04/resnets.html). Removing ReLU layer at the end of each residual unit, we observe a small but significant improvement in test performance and the converging progress becomes smoother. 
	```matlab
	res_cifar(20, 'modelType', 'resnet', 'reLUafterSum', false,...
		'expDir', 'data/exp/cifar-resNOrelu-20', 'gpus', [2])
	plot_results_mix('data/exp','cifar',[],[],'plots',{'resnet','resNOrelu'})
	```

	<div style="text-align:center"><img src ="https://raw.githubusercontent.com/zhanghang1989/ResNet-Matconvnet/master/figure/resnet_relu.png" width="420" /></div>	

0. **Imagenet2012.** download the dataset to `data/ILSVRC2012` and follow the instructions in `setup_imdb_imagenet.m`.
	```matlab
	run_experiments([50 101 152], 'gpus', [1 2 3 4 5 6 7 8]);
	```

0. **Your own dataset.** 
	```matlab
	run_experiments([18 34],'datasetName', 'minc',...
	'datafn', @setup_imdb_minc, 'nClasses', 23, 'gpus', [1 2]);
	```

### Fine-tune Your Own

0. **Download** 
	- the models to `data/models` : [imagenet-resnet-50-dag](http://www.vlfeat.org/matconvnet/pretrained) 
, [imagenet-resnet-101-dag](http://www.vlfeat.org/matconvnet/pretrained) 
, [imagenet-resnet-152-dag](http://www.vlfeat.org/matconvnet/pretrained) 
	- the datasets to `data/` : Material in Context Database [(minc)](http://opensurfaces.cs.cornell.edu/publications/minc/)

0. **Fine-tuning**
	```matlab
	res_finetune('datasetName', 'minc', 'datafn',...
	@setup_imdb_minc, 'gpus',[1 2]);
	```

### Changes
0. 06/21/2016:
	- Support Pre-activation model described in [Identity Mappings in Deep Residual Networks, Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](https://arxiv.org/abs/1603.05027)
0. 05/17/2016: 
	- Reproducing the experiments in Facebook [blog](http://torch.ch/blog/2016/02/04/resnets.html), removing ReLU layer at the end of each residual unit.
0. 05/02/2016: 
	- Supported **official Matconvnet** version.
	- Added Cifar experiments and plots.

0. 04/27/2016: Re-implementation of Residual Network:
	- The code benefits from Hang Su's [implementation](https://github.com/suhangpro/matresnet). 
	- The generated models are compatitible with [the converted models](http://www.vlfeat.org/matconvnet/pretrained). 
