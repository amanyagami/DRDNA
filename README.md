
# DrDNA: Detecting and Mitigating Soft Errors in Deep Neural Networks

This repository contains an unofficial implementation of the paper **"DrDNA: A Detection and Mitigation Framework for Soft Errors in Deep Neural Networks"** published in the ACM Digital Library (paper](https://dl.acm.org/doi/pdf/10.1145/3620666.3651349)).

## Overview

DrDNA is a framework for detecting and mitigating soft errors in deep neural networks. Soft errors, such as silent data corruptions, pose a significant challenge to reliable deep learning model deployment, particularly in safety-critical applications. DrDNA identifies soft errors through model behavior and implements techniques to mitigate the impact of these errors.


# Assumptions
Used hyperparameters Lambda1,Lambda2,Lambda3 = 1

## Profiling 
Dataset = Cifar 10 
Model  = Resnet 18 

Used 1000 samples for profiling. 

# Detection Results
100% sdc detected
Successfully able to replicate the results in the paper. 

Fault injected using custom fault injector based on pytorchFI. 
# Abnormality scores analysis 


![alt text](https://github.com/amanyagami/DRDNA/blob/main/output.png))

The affect of SDC increases the abnormality score of the values. 
 95% TPR threshold = 499. 
