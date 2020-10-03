# Session 9 - Data Augmentations

[![Open Jupyter Notebook](Images/nbviewer_badge.png)](https://nbviewer.jupyter.org/github/anubhabPanda/TSAI-EVA5/blob/master/Week9/EVA5S9.ipynb)

## Assignment Objective

* Move Week8's code transformations to Albumentations. Apply ToTensor, HorizontalFlip, Normalize (at min) + More (for additional points)
* Please make sure that your test_transforms are simple and only using ToTensor and Normalize
* Implement GradCam function as a module.
* Final code (notebook file) must use imported functions to implement transformations and GradCam functionality
* Target Accuracy is 87%

## Model Summary

The model reaches a final test accuracy of **88.39%** in **CIFAR-10** dataset in 20 epochs.
The model has **11,173,962 parameters**.

## Model Parameters

![architecture](Images/architecture.PNG)

## Hyperparameters

* Loss Function: Cross Entropy Loss
* Optimizer: SGD
* Learning Rate: 0.01
* Batch Size: 32
* Epochs: 20
* LR Scheduler: Step LR with step size of 6 and gamma of 0.1

## Transformations
* Rotate
* Horizontal Flip
* Cutout
  
## Variation of training and validation loss and accuracy with epochs

<img src="Images/Loss.PNG" width="400px" height='350' float='left'>
<img src="Images/Accuracy.PNG" width="400px" height='350' float='right'>

## Top Misclassified images
<img src="Images/Misclassified.PNG" width = "800px">

## Gradcam for some of the images in the validation set
<img src="Images/gradcam_1.PNG" width = "800px">
<img src="Images/gradcam_2.PNG" width = "800px">