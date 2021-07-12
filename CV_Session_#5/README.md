<br/>
<h1 align="center">Session 5: Coding Drill Down
<br/>

<!-- toc -->
### Contributors

| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\
<br>

## Objective:

Write a neural network to predict ***MNIST dataset*** with the following limitations
- 99.4% (this must be consistently shown in your last few epochs, and not a one-time achievement)
- Less than or equal to 15 Epochs
- Less than 10000 Parameters.

## Model Initialization:

1. #### HYPERPARAMS:

    num_epochs = 20<br>
    batch_size = 128<br>
    lr = 0.01<br>

2. #### CONSTANTS:

    NUM_CLASSES = 10<br>
    DATA_MEAN = (0.1307,)<br>
    DATA_STD = (0.3081,)<br >



| <centre>Model</centre> | <centre>Target</centre> |<centre>Results</centre> | <centre>Analysis</centre> | <centre> links </centre> |
| ------------ | ------------- | ---------- | --- | ------ |
| MNIST_basic_skeleton | 1. This skeleton of the model is built to check model performance<br>2. No Batch normalisation, dropout and image augmentations used<br>3. No lr_scheduler were used<br>| 1. Train Accuracy - 98.26<br>2. Test Accuracy - 97.96<br> 3. Total parameters - 10,578 | 1. The Model was learning based on the accuracy<br>2. There is also a slight overfit in model<br>3. Adding regularisation and scheduer to impore performance<br> | [Notebook link](./Session5/Notebooks/MNIST_basic.ipynb)<br>|
| MNIST_Regularization | 1. Model should achieve the target of 99.4 within 10k parameters<br>2. To achieve this we added dropout, lr_scheduler, batchnorm<br>3. In the basic skeleton the parameter count was little above 10k and this was reduced under 10k | 1. Train accuracy - 99.39<br>2. Test accuracy - 99.33<br>3. Total parameters - 8178 | 1. The performance the model reached 99.39<br>2. Overfitting of training data is minimised<br>3. Multiple set of dropout and lr_scheduler values were used<br>4. Dropout of `0` gave better accuracy<br>5. StepLR with step_size of 5 gave better accuracy | [Notebook link](./Session5/Notebooks/MNIST_regularization.ipynb)<br>|
| MNIST_augmentation | 1. Model parameters were further reduced below 8k<br>2. Basic Image agumentation were used<br>3. Plotted some incorrect predicted to check was went wrong in previous model | 1. Train accuracy - 98.23<br>2. Test accuracy - 99.44<br>3. Total parameters - 7946|1. The model is underfitting as we have added rotation to train data<br>2. RandomAffine with 7 degree orientation is introduced<br>| [Notebook link](./Session5/Notebooks/MNIST_augmetation.ipynb)<br>|

### Final Model Architecture:


    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 12, 26, 26]             108
                  ReLU-2           [-1, 12, 26, 26]               0
           BatchNorm2d-3           [-1, 12, 26, 26]              24
                Conv2d-4           [-1, 16, 24, 24]           1,728
                  ReLU-5           [-1, 16, 24, 24]               0
           BatchNorm2d-6           [-1, 16, 24, 24]              32
                Conv2d-7            [-1, 6, 24, 24]              96
                  ReLU-8            [-1, 6, 24, 24]               0
           BatchNorm2d-9            [-1, 6, 24, 24]              12
            MaxPool2d-10            [-1, 6, 12, 12]               0
               Conv2d-11           [-1, 12, 10, 10]             648
                 ReLU-12           [-1, 12, 10, 10]               0
          BatchNorm2d-13           [-1, 12, 10, 10]              24
               Conv2d-14             [-1, 14, 8, 8]           1,512
                 ReLU-15             [-1, 14, 8, 8]               0
          BatchNorm2d-16             [-1, 14, 8, 8]              28
               Conv2d-17             [-1, 14, 6, 6]           1,764
                 ReLU-18             [-1, 14, 6, 6]               0
          BatchNorm2d-19             [-1, 14, 6, 6]              28
               Conv2d-20             [-1, 14, 6, 6]           1,764
                 ReLU-21             [-1, 14, 6, 6]               0
          BatchNorm2d-22             [-1, 14, 6, 6]              28
    AdaptiveAvgPool2d-23             [-1, 14, 1, 1]               0
               Linear-24                   [-1, 10]             150
    ================================================================
    Total params: 7,946
    Trainable params: 7,946
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 0.55
    Params size (MB): 0.03
    Estimated Total Size (MB): 0.59
    ----------------------------------------------------------------

### Training Log:

```
  0%|          | 0/469 [00:00<?, ?it/s]EPOCH: 1
Loss=0.2920304238796234 Batch_id=468 Accuracy=83.17: 100%|██████████| 469/469 [00:15<00:00, 30.29it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0678, Accuracy: 9804/10000 (98.04%)

EPOCH: 2
Loss=0.1099965050816536 Batch_id=468 Accuracy=95.32: 100%|██████████| 469/469 [00:15<00:00, 29.66it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0540, Accuracy: 9841/10000 (98.41%)

EPOCH: 3
Loss=0.05505436658859253 Batch_id=468 Accuracy=96.20: 100%|██████████| 469/469 [00:16<00:00, 29.14it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0427, Accuracy: 9861/10000 (98.61%)

EPOCH: 4
Loss=0.0559239387512207 Batch_id=468 Accuracy=96.80: 100%|██████████| 469/469 [00:16<00:00, 27.97it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0340, Accuracy: 9896/10000 (98.96%)

EPOCH: 5
Loss=0.11785978823900223 Batch_id=468 Accuracy=97.18: 100%|██████████| 469/469 [00:16<00:00, 27.95it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0314, Accuracy: 9900/10000 (99.00%)

EPOCH: 6
Loss=0.1042698547244072 Batch_id=468 Accuracy=97.30: 100%|██████████| 469/469 [00:15<00:00, 29.49it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0303, Accuracy: 9902/10000 (99.02%)

EPOCH: 7
Loss=0.23318630456924438 Batch_id=468 Accuracy=97.31: 100%|██████████| 469/469 [00:15<00:00, 29.62it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0270, Accuracy: 9915/10000 (99.15%)

EPOCH: 8
Loss=0.1662347912788391 Batch_id=468 Accuracy=97.55: 100%|██████████| 469/469 [00:15<00:00, 29.70it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0270, Accuracy: 9916/10000 (99.16%)

EPOCH: 9
Loss=0.03322969749569893 Batch_id=468 Accuracy=97.44: 100%|██████████| 469/469 [00:15<00:00, 30.31it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0274, Accuracy: 9908/10000 (99.08%)

EPOCH: 10
Loss=0.11530936509370804 Batch_id=468 Accuracy=97.97: 100%|██████████| 469/469 [00:15<00:00, 29.92it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0205, Accuracy: 9938/10000 (99.38%)

EPOCH: 11
Loss=0.06971058994531631 Batch_id=468 Accuracy=98.10: 100%|██████████| 469/469 [00:15<00:00, 30.07it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0201, Accuracy: 9932/10000 (99.32%)

EPOCH: 12
Loss=0.05070294439792633 Batch_id=468 Accuracy=98.13: 100%|██████████| 469/469 [00:15<00:00, 30.08it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0196, Accuracy: 9938/10000 (99.38%)

EPOCH: 13
Loss=0.07360347360372543 Batch_id=468 Accuracy=98.24: 100%|██████████| 469/469 [00:15<00:00, 30.21it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0197, Accuracy: 9937/10000 (99.37%)

EPOCH: 14
Loss=0.02592841349542141 Batch_id=468 Accuracy=98.20: 100%|██████████| 469/469 [00:15<00:00, 30.38it/s]
  0%|          | 0/469 [00:00<?, ?it/s]
Test set: Average loss: 0.0193, Accuracy: 9940/10000 (99.40%)

EPOCH: 15
Loss=0.019798798486590385 Batch_id=468 Accuracy=98.23: 100%|██████████| 469/469 [00:15<00:00, 29.99it/s]

Test set: Average loss: 0.0190, Accuracy: 9944/10000 (99.44%)

```

### Model Performance:

<p align="center">
  <img src="./Session5/images/basic.png" width="800" title="hover text">
</p>
