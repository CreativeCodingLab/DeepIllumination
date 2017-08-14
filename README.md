# DeepRendering

[Video](https://www.youtube.com/watch?v=z_zmRWxU-PY)

## Introduction

Animation movie studios like Pixar uses a technique called Pathtracing which produces high-quality photorealistic images. Due to the computational complexity of this approach, it will take 8-16 hours to render depending on the composition of the scene. This time-consuming process makes Pathtracing unsuitable for interactive image synthesis. To achieve this increased visual quality in a real time application many approaches have been proposed in the recent past to approximate global illumination effects like ambient occlusion, reflections, indirect light, scattering, depth of field, motion blur and caustics. While these techniques improve the visual quality, the results are incomparable to the one produce by Pathtracing. We propose a novel technique where we make use of a deep generative model to generate high-quality photorealistic frames from a geometry buffer(G-buffer). The main idea here is to train a deep convolutional neural network to find a mapping from G-buffer to pathtraced image of the same scene. This trained network can then be used in a real time scene to get high-quality results.


#### Table of Contents

* [Installation](#installation)
* [Running](#running)
* [Dataset](#dataset)
* [Hyperparameters](#hyperparameter)
* [Results](#results)
* [Improvements](#improvements)
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

Run python train.py --dataset dataset/[name of your dataset]/ --n_epochs num of epochs

python train.py --dataset dataset/DeepRendering --n_epochs 200

check train.py for more options.

Validation is done afer every epoch and will be inside validation/DeepRendering/

To test, 

Run python test.py --dataset dataset/[name of your dataset]/ --model checkpoint/[name of your checkpoint]

python train.py --dataset dataset/DeepRendering --model checkpoint/

Check results/DeepRendering for the output.

## Dataset
We picked random 40 images from pixar movies, added gaussian noise of different standard deviation, 5 sets of 5 different standard deviation making a total of 1000 images for the training set. For validation we used 10 images completely different from the training set and added gaussian noise. For testing we had both added gaussian images and real noisy images.
