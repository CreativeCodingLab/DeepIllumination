# DeepRendering

[Video](https://www.youtube.com/watch?v=z_zmRWxU-PY)

## Introduction

Animation movie studios like Pixar uses a technique called Pathtracing which produces high-quality photorealistic images. Due to the computational complexity of this approach, it will take 8-16 hours to render depending on the composition of the scene. This time-consuming process makes Pathtracing unsuitable for interactive image synthesis. To achieve this increased visual quality in a real time application many approaches have been proposed in the recent past to approximate global illumination effects like ambient occlusion, reflections, indirect light, scattering, depth of field, motion blur and caustics. While these techniques improve the visual quality, the results are incomparable to the one produce by Pathtracing. We propose a novel technique where we make use of a deep generative model to generate high-quality photorealistic frames from a geometry buffer(G-buffer). The main idea here is to train a deep convolutional neural network to find a mapping from G-buffer to pathtraced image of the same scene. This trained network can then be used in a real time scene to get high-quality results.
