# ResNet-Matconvnet

This repository is a Matconvnet re-implementation of ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385). You can train Deep Residual Network on ImageNet from Scratch or fine-tune pre-trained model or your own dataset. It This repo is created by [Hang Zhang](http://www.hangzh.com).

### Table of Contents
0. [Get Started](#get-started)
0. [Train from Scratch](#train-from-Scratch)
0. [Fine-tune Your Own](#fine-tune-your-own)

### Get Started

The code relies on [vlfeat](http://www.vlfeat.org/), and [a modified version of matconvnet](https://github.com/zhanghang1989/matconvnet), which should be downloaded and built before running the experiments. You can use the following commend to download them.
	
	git clone --recurse-submodules https://github.com/zhanghang1989/ResNet-Matconvnet.git

### Train from Scratch
**1. Download Imagenet2012 dataset** to `data/ILSVRC2012` and follow the instructions in `setup_imdb_imagenet.m`

**2. Usage example:** 
	
	run_experiments([18 34 50 101 152], 'gpus', [1 2 3 4 5 6 7 8]);

**3. On your own dataset:** 
	
	run_experiments([18 34 50 101],'datasetName',...
	'yourdata', 'datafn', @setup_imdb_generic, 'gpus', [1 2]);

For training ResNet on CIFAR dataset, please refer to Hang Su's [GitHub](https://github.com/suhangpro/matresnet).

### Fine-tune Your Own (Coming Soon)

**1. Download the models** to `data/models`, if you want to fine-tune a pre-trained RestNet      
  * [imagenet-resnet-50-dag](http://www.vlfeat.org/matconvnet/pretrained) 
  * [imagenet-resnet-101-dag](http://www.vlfeat.org/matconvnet/pretrained) 
  * [imagenet-resnet-152-dag](http://www.vlfeat.org/matconvnet/pretrained) 

**2. Download the datasets** to `data/`, if you want to fine-tune ResNet on them
  * Reflectance Disks [(reflectance)](http://hangzh.com/Software.html)  
  * Flickr Material Database [(fmd)](http://people.csail.mit.edu/celiu/CVPR2010/FMD/) 
  * Describable Textures Dataset [(dtd)](http://www.robots.ox.ac.uk/~vgg/data/dtd)
  * Textures under varying Illumination [(kth)](http://www.nada.kth.se/cvap/databases/kth-tips/)
  * Material in Context Database [(minc)](http://opensurfaces.cs.cornell.edu/publications/minc/)
