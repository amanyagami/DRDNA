
# DrDNA: Detecting and Mitigating Soft Errors in Deep Neural Networks

This repository contains my own implementation of the paper **"DrDNA: A Detection and Mitigation Framework for Soft Errors in Deep Neural Networks"** published in the ACM Digital Library (paper](https://dl.acm.org/doi/pdf/10.1145/3620666.3651349)) . 
(No official implementation) 

## Overview

DrDNA is a framework for detecting and mitigating soft errors in deep neural networks. Soft errors, such as silent data corruptions, pose a significant challenge to reliable deep learning model deployment, particularly in safety-critical applications. DrDNA identifies soft errors through model behavior and implements techniques to mitigate the impact of these errors.

## Methodology 

This is a post-hoc method and involves 2 major steps:
### Profiling - 
Each Layer in the model has k sites (cohort size)  as detection sites. 
these sites are used to generate there stats for the data Tau1,Tau2,Tau3

Tau1 - Each neuron in the k sites produces activation on inference. a histogram of activations is created for each such site in each layer

Tau2 - A histogram of activations of all the detection sites in a layer 

Tau3 - Neurons producing highest and lowest activations. 
### Scoring
These 3 profiles are used to generate 3 scores

Tau1 score - (1 - fi ) f1 is the frequency of the activation in the histogram of the site. (higher fi means higher frequency hence less abnormal) 

Tau2 score - EMD distance between layer histogram during inference and profiling . (high EMD means more abnormal)

Tau3 score - 0 if neuron are same , 0.5 if one neuron is same. 1 if both neurons are different

Score[layer] = lambda1 * tau1 + lambda2 * tau2 + lambda3 * tau3 + Score[layer-1]
###  Detection

The scores are generated for data without fault.

Since we have the layer-wise scores for data without fault during profiling we can use them for early detection. 

T` score for layer j (with fault)

T score for layer j (no fault, available after profiling) 

Margin = (T` - T)/ T

Margin can be configured experminatally. 
if Margin > thresold SDC Detected!
else not detected.

# Assumptions
Used hyperparameters Lambda1 ,Lambda2 ,Lambda3  = 1

##
Dataset = Cifar 10 

Model  = Resnet 18 

Used 1000 samples for profiling. 

# Detection Results
100% SDC detected

Successfully able to replicate the results in the paper. 

Fault injected using custom fault injector based on pytorchFI. 
# Abnormality scores analysis 


![alt text](https://github.com/amanyagami/DRDNA/blob/main/output.png))

The affect of SDC increases the abnormality score of the values. 
 95% TPR threshold = 499. 
