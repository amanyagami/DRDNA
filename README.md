
# DrDNA: Detecting and Mitigating Soft Errors in Deep Neural Networks

This repository contains an unofficial implementation of the paper **"DrDNA: A Detection and Mitigation Framework for Soft Errors in Deep Neural Networks"** published in the ACM Digital Library (paper](https://dl.acm.org/doi/pdf/10.1145/3620666.3651349)).

## Overview

DrDNA is a framework for detecting and mitigating soft errors in deep neural networks. Soft errors, such as silent data corruptions, pose a significant challenge to reliable deep learning model deployment, particularly in safety-critical applications. DrDNA identifies soft errors through model behavior and implements techniques to mitigate the impact of these errors.

## Features

- **Soft Error Detection:** Automatically identifies silent data corruptions (SDC) and other soft errors in model computations.
- **Mitigation Techniques:** Implements state-of-the-art techniques to reduce the impact of detected errors.
- **Model Compatibility:** Works with various deep learning architectures, such as ResNet, AlexNet, DenseNet, and VGG.

## Profiling 
Dataset = Cifar 10 
Model  = Resnet 18 

Fault injected using custom fault injector based on pytorchFI. 

Results : 
100% sdc detected
Successfully able to replicate the results in the paper. 
