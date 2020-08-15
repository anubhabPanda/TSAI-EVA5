# Week 4- Architectural Basics

[![Open Jupyter Notebook](Images/nbviewer_badge.png)](https://nbviewer.jupyter.org/github/anubhabPanda/TSAI-EVA5/blob/master/Week4/S4_Assignment_Solution.ipynb)

## Assignment Objective: 
The goal of this assignment is to achieve **>99.40%** accuracy on the test set of the **MNIST handwritten dataset**. The model needs to have the following constraints :

* Less than 20K Parameters
* Less than 20 epochs
* No fully connected layers

## Network Architecture



## Approach

* Performed three 3X3 convolutions before maxpooling to achieve a receptive field of 7 since for images of size 28X28, edges and gradients are at a minimum size of 7 pixels.
* Used Batchnorm and Dropout of 10% after every convolutional block except the last block.
* Used Global Average Pooling at 8X8 to c 
## Network Parameters

![](Images/Network_Parameters.PNG)

Total number of parameters in the network is 12,000.
