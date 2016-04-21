# ResNet-Matconvnet (Coming Soon)
Created by [Hang Zhang](www.hangzh.com)

This repo trains or fine-tunes Deep Residual Network on ImageNet or your own dataset. ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385). 

### Get Started

The code relies on [vlfeat](http://www.vlfeat.org/), and [a modified version of matconvnet](https://github.com/zhanghang1989/matconvnet), which should be downloaded and built before running the experiments. You can use the following commend to install them.


### Train ResNet on ImageNet
**Download Imagenet2012 dataset** to `data/ILSVRC2012` and follow the instructions in `setup_imdb_imagenet.m`

**Usage example:** 
	
	run_experiments([18 34 50 101 152], 'gpus', [1 2]);

**On your own dataset:** 
	
	run_experiments([18 34 50 101],'datasetName',...
	'reflectance', 'datafn', @setup_imdb_reflectance, 'gpus', [1 2]);

For training ResNet on CIFAR dataset, please refer to Hang Su's [GitHub](https://github.com/suhangpro/matresnet).

### Fine-tune on Your Own Datasets

* Download one of the following models to `data/models`, if you want to fine-tune a pre-trained RestNet      
    * [imagenet-resnet-50-dag](http://www.vlfeat.org/matconvnet/pretrained) 
    * [imagenet-resnet-101-dag](http://www.vlfeat.org/matconvnet/pretrained) 
    * [imagenet-resnet-152-dag](http://www.vlfeat.org/matconvnet/pretrained) 

* Download the following material datasets to `data/`, if you want to fine-tune ResNet on them
    * Reflectance Disks [(reflectance)](https://goo.gl/6Kwg13)  
    * Flickr Material Database [(fmd)](http://people.csail.mit.edu/celiu/CVPR2010/FMD/) 
    * Describable Textures Dataset [(dtd)](http://www.robots.ox.ac.uk/~vgg/data/dtd)
    * Textures under varying Illumination [(kth)](http://www.nada.kth.se/cvap/databases/kth-tips/)

### Acknowldgements

We thank [vlfeat](http://www.vlfeat.org/) and [matconvnet](http://www.vlfeat.org/matconvnet) teams for creating and maintaining these excellent packages. Thank Hang Su for [MatConvNet Implementation of ResNet](https://github.com/suhangpro/matresnet).
