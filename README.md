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
git clone --recurse-submodules https://github.com/zhanghang1989/ResNet-Matconvnet.git
```
If you have problem with compiling, please refer to the [link](http://zhanghang1989.github.io/ResNet/#imagenet).	
### Train from Scratch
0. **Cifar.** Reproducing Figure 6 from the original paper.
	```matlab
	run_cifar_experiments([20 32 44 56 110], 'plain', 'gpus', [1]);
	run_cifar_experiments([20 32 44 56 110], 'resnet', 'gpus', [1]);
	```
	
	<div style="text-align:center"><img src ="https://raw.githubusercontent.com/zhanghang1989/ResNet-Matconvnet/master/figure/plain_cifar.png" width="420" /><img src ="https://raw.githubusercontent.com/zhanghang1989/ResNet-Matconvnet/master/figure/resnet_cifar.png" width="420" /></div>	

	<p style="text-align:center; font-size:75%; font-style: italic;">Cifar Experiments</p>
	
0. **Imagenet2012.** download the dataset to `data/ILSVRC2012` and follow the instructions in `setup_imdb_imagenet.m`.
	```matlab
	run_experiments([18 34 50 101 152], 'gpus', [1 2 3 4 5 6 7 8]);
	```

0. **Your own dataset.** 
	```matlab
	run_experiments([18 34 50 101],'datasetName',...
	'yourdata', 'datafn', @setup_imdb_generic, 'gpus', [1 2]);
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
0. 05/02/2016: 
	- Supported **official Matconvnet** version.
	- Added Cifar experiments and plots.

0. 04/27/2016: Re-implementation of Residual Network:
	- The code benefits from Hang Su's [implementation](https://github.com/suhangpro/matresnet). 
	- Added ImageNet experiments. The models are compatitible with [the converted models](http://www.vlfeat.org/matconvnet/pretrained). 
