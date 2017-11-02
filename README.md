# DeepIllumination (More datasets coming soon)| [Video](https://www.youtube.com/watch?v=z_zmRWxU-PY)

<img src="https://github.com/CreativeCodingLab/DeepRendering/blob/master/teaser.png" alt="alt text" width="945" height="322.5">

#### A paper introducing our DeepIllumination approach is currently in preparation to EuroGraphics. Pre-print is available on arXiv at https://arxiv.org/abs/1710.09834

## Introduction

Animation movie studios like Pixar uses a technique called Pathtracing which produces high-quality photorealistic images. Due to the computational complexity of this approach, it will take 8-16 hours to render depending on the composition of the scene. This time-consuming process makes Pathtracing unsuitable for interactive image synthesis. To achieve this increased visual quality in a real time application many approaches have been proposed in the recent past to approximate global illumination effects like ambient occlusion, reflections, indirect light, scattering, depth of field, motion blur and caustics. While these techniques improve the visual quality, the results are incomparable to the one produce by Pathtracing. We propose a novel technique where we make use of a deep generative model to generate high-quality photorealistic frames from a geometry buffer(G-buffer). The main idea here is to train a deep convolutional neural network to find a mapping from G-buffer to pathtraced image of the same scene. This trained network can then be used in a real time scene to get high-quality results.


#### Table of Contents

* [Installation](#installation)
* [Running](#running)
* [Dataset](#dataset)
* [Hyperparameters](#hyperparameter)
* [Results](#results)
* [Notes](#notes)
* [Credits](#credits)

## Installation

To run the project you will need:
 * python 3.5
 * pytorch
 * [CHECKPOINT FILE](https://uofi.box.com/v/DeepRenderingCheckpointFile)
 * [Dataset](https://uofi.box.com/v/DeepRenderingDataset)

## Running

Once you have all the depenedencies ready, do the folowing:
Download the dataset and extract it.
Download the checkpoint file and extract it.
Now you will have two folders checkpoint and dataset.

To train, move your training set to dataset/[name of your dataset]/train and validation set to dataset/[name of your dataset]/val
```
Run python train.py --dataset dataset/[name of your dataset]/ --n_epochs num of epochs
```
```
python train.py --dataset dataset/DeepRendering --n_epochs 200
```
check train.py for more options.
Validation is done afer every epoch and will be inside validation/DeepRendering/

To test, 
Run 
``` 
python test.py --dataset dataset/[name of your dataset]/ --model checkpoint/[name of your checkpoint] 
```
``` 
python train.py --dataset dataset/DeepRendering --model checkpoint/
```
Check results/DeepRendering for the output.

## Dataset
Dataset was created using a simple cornell box made with Unity3D. GBuffers(depth, normal, albedo, direct light) and VXGI outputs are extracted.

For training, we used 4 object - sphere, cylinder, cube, stanford bunny and a dragon.

For testing we used 2 object - statue and buddha

## Hyperparameters
* Number of epochs - 12
* L1 factor - 100
* lr - 0.0002
* n_filters - 128

## Results

<img src="https://github.com/CreativeCodingLab/DeepRendering/blob/master/output.png" alt="alt text" width="960" height="480">

 ## Notes
 
* One network for one world - training a network by introducing it to the rules of one enviornment produce good result rather than to make a generalized network to work for all kinds of enviornment. Also we only need fewer parameters.
* Network gives a good approximatation even though it has only seen the training set 12 times.
* Our network failed to learn ambient occlsuion. We believe with more dataset focusing on ambient occlusion we can force the network to learn AO.

 ## Credits
* [Imgae to Image Translation](https://arxiv.org/abs/1611.07004)
* [SRGAN](https://arxiv.org/pdf/1609.04802.pdf)
* [Creating photorealistic images from gameboy camera](http://www.pinchofintelligence.com/photorealistic-neural-network-gameboy/)
* [CS231n](https://cs231n.github.io/)


