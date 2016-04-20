# ResNet-Matconvnet (Coming Soon Later Today)
Created by [Hang Zhang](www.hangzh.com)

This repo trains or fine-tunes Deep Residual Network on ImageNet or your own dataset. ["Deep Residual Learning for Image Recognition",Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun](http://arxiv.org/abs/1512.03385). 

### Get Started

* The code relies on [vlfeat](http://www.vlfeat.org/), and a modified version of [matconvnet](https://github.com/zhanghang1989/matconvnet-18), which should be downloaded and built before running the experiments. You can use the following commend to install them.

* Download one of the following models to `data/models`, if you want to fine-tune a pre-trained RestNet      
    1. [imagenet-resnet-50-dag](http://www.vlfeat.org/matconvnet/pretrained) 
    2. [imagenet-resnet-101-dag](http://www.vlfeat.org/matconvnet/pretrained) 
    3. [imagenet-resnet-152-dag](http://www.vlfeat.org/matconvnet/pretrained) 

* Download the following material datasets to `data/`, if you want to train or fine-tune ResNet on them
    * Reflectance Disks [(reflectance)](https://goo.gl/6Kwg13)  
    * Flickr Material Database [(fmd)](http://people.csail.mit.edu/celiu/CVPR2010/FMD/) 
    * Describable Textures Dataset [(dtd)](http://www.robots.ox.ac.uk/~vgg/data/dtd)
    * Textures under varying Illumination [(kth)](http://www.nada.kth.se/cvap/databases/kth-tips/)

### Train from Scratch 

For training ResNet on CIFAR dataset, please refer to Hang Su's [GitHub](https://github.com/suhangpro/matresnet).

### Fine-tune on ImageNet or Other Datasets

### Acknowldgements

We thank [vlfeat](http://www.vlfeat.org/) and [matconvnet](http://www.vlfeat.org/matconvnet) teams for creating and maintaining these excellent packages. Thank Hang Su for [MatConvNet Implementation of ResNet](https://github.com/suhangpro/matresnet).
