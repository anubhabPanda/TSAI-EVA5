# Week 5- Coding Drill DOwn

## Assignment Objective

The goal of this assignment is to achieve **>99.40%** accuracy on the test set of the **MNIST handwritten dataset**. The objective needs to be achieved using following conditions :

* Consistently reach greater than or equal to 99.4% accuracy in the last few epochs
* Less than 10K Parameters
* Less than or equal to 15 epochs
* The objective is to be reached in exactly 4 steps
* Each step needs to have a "target, result, analysis" TEXT block (either at the start or the end)
* Each step must be convincing as to why it was  decided that the target should be what it was decided to be, and the analysis must be correct

## Step1 (Setup and Skeleton)

[![Open Jupyter Notebook](Images/nbviewer_badge.png)](https://nbviewer.jupyter.org/github/anubhabPanda/TSAI-EVA5/blob/master/Week5/Step1.ipynb)

### Target

1. Get the setup right
2. Read MNIST dataset, set train test split and create Data Loader
3. Get the summary statistics for the data
4. Set initial transforms and apply transformation to the train and test set separately
5. Get the basic neural net architecture skeleton right. We will try and avoid changing the skeleton later
6. Set basic training and test loop

### Results

1. Parameters: 992,800
2. Best Training Accuracy: 99.88
3. Best Test Accuracy: 99.14

### Analysis

1. Very heavy model for such an easy problem. Lots of parameters. Have to reduce the number of parameters in the next step
2. Test accuracy is way below the target accuracy
3. Model is overfitting

## Step2 (Reducing the number of parameters)

[![Open Jupyter Notebook](Images/nbviewer_badge.png)](https://nbviewer.jupyter.org/github/anubhabPanda/TSAI-EVA5/blob/master/Week5/Step2.ipynb)

### Target

1. Make the model lighter by reducing the number of parameters
    * Reduce the number of channels
    * Use Global Average Pooling instead of a large 7X7 kernel in the output block
2. Have more number of channels before maxpooling
3. Transition layer after reaching a receptive field of 5 instead of 7 since in smaller images, edges and gradients starts getting detected at a receptive field of 5.

### Results

1. Parameters: 9,712
2. Best Training Accuracy: 98.44
3. Best Test Accuracy: 98.53

### Analysis

1. Good model
2. Model is not overfitting
3. There is room for improvement in terms of accuracy. Can push the model further.

## Step3 (Improving model performance by adding Batch Norm and regularization to prevent overfitting)

[![Open Jupyter Notebook](Images/nbviewer_badge.png)](https://nbviewer.jupyter.org/github/anubhabPanda/TSAI-EVA5/blob/master/Week5/Step3.ipynb)

### Target

1. Apply Batch Norm after every convolution layer except the last layer before output to improve accuracy 
2. Apply dropouts if there is overfitting after applying batch norm

### Results

1. Parameters: 9,904
2. Best Training Accuracy: 98.75
3. Best Test Accuracy: 99.24

### Analysis

1. Good model but still we haven't been able to reach our target accuracy
2. Applied low dropout to prevent overfitting and for model stability
3. We can still increase the accuracy further
4. We can't increase the model capacity any further. We have to rely on some other method to increase the accuracy further
5. We can decrease batch size and try image augmentations

## Step4 (Improving model performance by adding Image Augmentation and LR Scheduler)

[![Open Jupyter Notebook](Images/nbviewer_badge.png)](https://nbviewer.jupyter.org/github/anubhabPanda/TSAI-EVA5/blob/master/Week5/Step4.ipynb)

### Target

1. Apply Image Augmentation techniques like Random Rotation since there are some images which are slightly rotated
2. Apply LR scheduler for finding good Learning rate for better convergence
3. Reduce batch size for more parameter updates

### Results

1. Parameters: 9,904
2. Best Training Accuracy: 99.08
3. Best Test Accuracy: 99.44

### Analysis

1. Very good model. Able to achieve the target accuracy with required parameters
2. Adding Image Augmentation and LR scheduler have made the model more robust. There is less variance in the validation accuracy
