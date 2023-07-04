# WL-DeepGCN

This is a Pytorch implementation of Deep Graph Convolutional Networks combined with weight-learning, as described in our paper.

## Requirement

`Python 3.7.12`

`Pytorch 1.13.0`

`Cuda 11.6`

`Numpy 1.21.6`

`scikit-learn 1.0.2`

`Nilearn 0.9.2`

This code has been tested using Pytorch on a GTX 3060 GPU.

## Dataset

Before training, you need to get the data on [ABIDE](http://fcon_1000.projects.nitrc.org/indi/abide/) first. Please change the path in the code to your path.

## Training

`python Nested_CV.py --train=1`

You can also evaluate directly with the `pth` we provided.

`python Nested_CV.py --train=0`

## Innovative features

1. We propose a new DeepGCN framework for ASD diagnosis using brain functional networks for classification. A weight-learning network automatically exploits the pairwise associations of non-imaging data in the latent space for constructing graph edge weights, building an adaptive population graph model with variable edges. The deep GCN can better represent the connections between different topics for semi-supervised learning. Our method not only achieves end-to-end training for ASD diagnosis but can also be extended to the diagnosis of other psychiatric disorders.


2. We propose the residual connection of a graph convolutional neural network to avoid the problems of the DeepGCN gradient explosion and gradient disappearance. The residual unit is able to reduce the feature information overfitting problem caused by the convolution operation, and it is used to concatenate the output of this layer and the previous layer to prepare for the input of the next layer.


3. We introduce an EdgeDrop strategy to avoid overfitting and oversmoothing problems during DeepGCN training. Random edge dropping in the raw graph during model training can make the node  onnections sparser. This reduces the oversmoothing aggregation speed and reduces subsequent information loss.

## The Important explanatory comments

1. Our proposed method appears to contain many steps, but most of them are improvements in graph construction. Our improvements to DeepGCN are clear and concise, and have been validated in previous work. The use of a simple structure can enhance the expressive power of DeepGCN by inputting more valuable graph structures into the GCN, which is evident from our code and experimental results.
2. Differences from previous work: In the Weighted-Learning Network (WL), our inspiration comes from Huang et al.'s Pairwise Association Encoder (PAE), but the structure of WL is completely different from that of PAE. The PAE uses a fixed number of hidden layers with a multilayer perceptron (MLP) structure, while WL uses up to three adaptive fully connected layers to learn the potential connectivity between non-imaging data. This is beneficial for extracting high-level features from low-level features and making more effective use of the value of non-imaging data. Since the non-imaging data has a uniform format, we added a Dropout layer after each fully connected layer to reduce the co-adaptation between neurons and limit model complexity to avoid overfitting issues.