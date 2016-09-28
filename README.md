# ResNet-Matconvnet on MNIST dataset

This repository is a clone from the original source (https://github.com/zhanghang1989/ResNet-Matconvnet) by zhanghang (https://github.com/zhanghang1989) - For my personal understanding and experiments, I haved added tests for mnist dataset. If you are new to this repository, please visit the original source for more understanding about using the code.

I have added two files res_mnist.m and run_mnist_expriments.m to the repo. run_mnist_experiments.m will take care of running the experiments on Mnist dataset once you have correctly set the paths to mnist dataset, and you have added VLFEAT and MATCONVNET sources to the dependencies folder.

## Test on MNIST dataset
The mnist dataset can be downloaded from one of my other repositories here (https://github.com/wajihullahbaig/DeepNeuralNetworks/blob/master/data/mnist_uint8.mat). 
Three different experiments have been conducted. In each experiments, 20 and 32 layers have been used to produce the results.
The main purpose was to understand training own dataset and view the behaviour these networks. 
	

	1. Plain nets [20 32]
	2. Residual nets [20 32]
	3. Residual nets with now ReLU [20 32]

## Results 

Epochs = 40, GPU Tests 

| Networks Type        | Train Error  | Test Error |
| ---------------------|:------------:| ----------:|
| Plain 20             | 0.0042       | 0.0394     | 
| Plain 32		       | 0.0046       | 0.1195     | 
| Residual 20		   | 0.0043       | 0.0581     | 
| Residual 32		   | 0.0039       | 0.0255     | 
| Residual-No-reLU 20  | 0.0057       | 0.0186	   | 
| Residual-No-reLU 32  | 0.0042       | 0.0170     | 
		         
### Plain
![Plain nets](https://github.com/wajihullahbaig/ResNet-Matconvnet/blob/master/figure/plain.jpg)

### Resnet 
![Residual nets](https://github.com/wajihullahbaig/ResMet-Matconvnet/blob/master/figure/resnet.jpg)

### Resnet-No-reLU 
![Residual-No-ReLU nets](https://github.com/wajihullahbaig/ResMet-Matconvnet/blob/master/figure/resnet-No-reLU.jpg)

## Platform/Dev Environmets/Dependencies etc

	1- Linux 16.04
	2- Matconvnet-1.0-beta20
	3- VLFeat
	4- Matlab 2015a
	5- Cuda 7.5
	6- Nvidia GT 430
