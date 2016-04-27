# ResNet-Matconvnet

This repository is a Matconvnet re-implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385). You can train Deep Residual Network on ImageNet from Scratch or fine-tune pre-trained model or your own dataset. This repo is created by [Hang Zhang](http://www.hangzh.com).

### Table of Contents
0. [Get Started](#get-started)
0. [Train from Scratch](#train-from-scratch)
0. [Fine-tune Your Own](#fine-tune-your-own)

### Get Started

The code relies on [vlfeat](http://www.vlfeat.org/), and [a modified version of matconvnet](https://github.com/zhanghang1989/matconvnet), which should be downloaded and built before running the experiments. You can use the following commend to download them.
```sh
git clone --recurse-submodules https://github.com/zhanghang1989/ResNet-Matconvnet.git
```
	
### Train from Scratch

0. **Imagenet2012.** download the dataset to `data/ILSVRC2012` and follow the instructions in `setup_imdb_imagenet.m`.
	```matlab
	run_experiments([18 34 50 101 152], 'gpus', [1 2 3 4 5 6 7 8]);
	```

0. **Your own dataset.** 
	```matlab
	run_experiments([18 34 50 101],'datasetName',...
	'yourdata', 'datafn', @setup_imdb_generic, 'gpus', [1 2]);
	```

### Fine-tune Your Own (Coming Soon)

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
0. 04/27/2016: The code benefits from Hang Su's [implementation](https://github.com/suhangpro/matresnet). This re-implementation contains:
	- support for ImageNet experiments and fine-tuning the pre-trained/converted models
	- our models are compatitible with [Matconvnet converted models](http://www.vlfeat.org/matconvnet/pretrained). 
	- batch normailization layer to the shortcuts between different dimensions (see analysis).

