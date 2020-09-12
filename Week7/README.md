# Session 7 - Advanced Convolutions

The model reaches a test accuracy of **84.83%** in **CIFAR-10** dataset. The model uses the following types of convolutions:

- 3x3 Convolution
- Pointwise Convolution
- Atrous Convolution
- Depthwise Separable Convolution
- Max Pooling

The model has **94,218 parameters**.

## Model Architecture

![architecture](images/architecture.png)

### Parameters and Hyperparameters

- Loss Function: Cross Entropy Loss
- Optimizer: SGD
- Learning Rate: 0.01
- Dropout Rate: 0.1
- Batch Size: 64
- Epochs: 50

## Change in Validation Loss and Accuracy

<img src="Images/Val_loss.png" width="450px">
<img src="Images/Val_Accuracy.png" width="450px">

## Top Misclassified images
<img src="Images/Top_Losses.png" width = "600px">