<br/>
<h1 align="center">Session 7: Advanced Concepts
<br/>
<!-- toc -->
    <br>
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)

#### Contributors

<p align="center"> <b>Team - 6</b> <p>
    
| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->
    
## Problem Statement 
        
To achieve 85% accuracy with total Params less than 200k in **`CIFAR10`** dataset and should use the following
    
```
1.  To use GPU
2.  To use architecture C1C2C3C40 (No MaxPooling, but 3 3x3 layers with stride of 2 instead)
3.  To use Dilated kernels here instead of MP or strided convolution    
4.  To achieve total Receptive Field more than 52
5.  To use Depthwise Separable Convolution at least in 2 of the layers 
6.  To use Dilated Convolution at least in one of the layers 
7.  To use GAP (compulsory mapped to # of classes):- CANNOT add FC after GAP to target #of classes
8.  To use correct Normalization values by having computing mean and std value in Transform
9.  To use albumentation library and apply:
    a. horizontal flip
    b. shiftScaleRotate 
    c. coarseDropout (max_holes = 1, max_height=16px, max_width=1, min_holes = 1, min_height=16px, 
    min_width=16px, fill_value=(mean of your dataset), mask_fill_value = None)
    d. grayscale
```    

### Model Achitecture:

[iNotebook link](./final_assignment7.ipynb)<br>
Model follows `C1C2C3C4` architecture, no maxpooling was performed <br>
Dilation and stride is performed to reduce the sapatial size of the channels <br>
The **`Receptive Field`** of the model after performing convolutions are **`147,978`** <br>
Model Uses Depthwise Sepearble convlution in 2 layersone at the end of `ConvBlock1` and another at `ConvBlock3`<br>
No Dropout or maxpooling were used in model building<br>
Model uses GAP and no FC layers are used in target prediction<br>
**`Total Params`** used by the model is `147,978`
Dilated convolution is just a convolution applied to input with defined gaps.<br>
Given an 2D input image, dilation rate k=1 is normal convolution and k=2 means skipping one pixel per input and k=4 means skipping 3 pixels. 
    <br>
    
    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 32, 32, 32]             896
                  ReLU-2           [-1, 32, 32, 32]               0
           BatchNorm2d-3           [-1, 32, 32, 32]              64
                Conv2d-4           [-1, 64, 32, 32]          18,496
                  ReLU-5           [-1, 64, 32, 32]               0
           BatchNorm2d-6           [-1, 64, 32, 32]             128
                Conv2d-7           [-1, 64, 32, 32]             640
                Conv2d-8           [-1, 64, 32, 32]           4,160
                  ReLU-9           [-1, 64, 32, 32]               0
          BatchNorm2d-10           [-1, 64, 32, 32]             128
               Conv2d-11           [-1, 64, 16, 16]          36,928
                 ReLU-12           [-1, 64, 16, 16]               0
          BatchNorm2d-13           [-1, 64, 16, 16]             128
               Conv2d-14           [-1, 32, 16, 16]           2,080
                 ReLU-15           [-1, 32, 16, 16]               0
          BatchNorm2d-16           [-1, 32, 16, 16]              64
               Conv2d-17           [-1, 32, 16, 16]           9,248
                 ReLU-18           [-1, 32, 16, 16]               0
          BatchNorm2d-19           [-1, 32, 16, 16]              64
               Conv2d-20           [-1, 64, 16, 16]          18,496
                 ReLU-21           [-1, 64, 16, 16]               0
          BatchNorm2d-22           [-1, 64, 16, 16]             128
               Conv2d-23           [-1, 64, 16, 16]             640
               Conv2d-24           [-1, 64, 16, 16]           4,160
                 ReLU-25           [-1, 64, 16, 16]               0
          BatchNorm2d-26           [-1, 64, 16, 16]             128
               Conv2d-27             [-1, 64, 8, 8]          36,928
                 ReLU-28             [-1, 64, 8, 8]               0
          BatchNorm2d-29             [-1, 64, 8, 8]             128
               Conv2d-30             [-1, 32, 8, 8]           2,080
                 ReLU-31             [-1, 32, 8, 8]               0
          BatchNorm2d-32             [-1, 32, 8, 8]              64
               Conv2d-33             [-1, 32, 8, 8]           9,248
                 ReLU-34             [-1, 32, 8, 8]               0
          BatchNorm2d-35             [-1, 32, 8, 8]              64
               Conv2d-36             [-1, 10, 8, 8]           2,890
                 ReLU-37             [-1, 10, 8, 8]               0
    AdaptiveAvgPool2d-38             [-1, 10, 1, 1]               0
    ================================================================
    Total params: 147,978
    Trainable params: 147,978
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 6.07
    Params size (MB): 0.56
    Estimated Total Size (MB): 6.65
    ----------------------------------------------------------------
    
#### Mean and Standard Deviation calculation:

Mean - μ and Standard deviation - σ of the dataset is calculated using the below function<br>
loader takes DataLoader type input and the num_channels, number of channels in the input image<br>
    
```bash
def get_mean_std(loader,num_channels):
    channels_sum, channels_sqrd_sum, num_batches = 0, 0, 0

    for data, _ in tqdm(loader):
        channels_sum += torch.mean(data, dim=[0, 2, num_channels])
        channels_sqrd_sum += torch.mean(data ** 2, dim=[0, 2, num_channels])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_sqrd_sum / num_batches - mean ** 2) ** 0.5

    return mean, std
```
📖 
[![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1hSEYabJtPm1OYK4N-v-OZ8Yn9bA_1cx9?authuser=2#scrollTo=ThrAy24XaSd9)


## Cutout and Augmentations:

For image augmentations, albumentation library is used. Albumentation is a Python library for fast and flexible [image augmenation](https://en.wikipedia.org/wiki/Data_augmentation)<br>
    
```bash
pip install -U albumentations
```
transformation performed on the data are
1. `HorizontalFlip` - Flip the input horizontally around the y-axis.
2. `ShiftScaleRotate` - Randomly apply affine transforms: translate, scale and rotate the input.
3. `Normalizee` - Normalization is applied by the formula: **img = (img - mean * max_pixel_value) / (std * max_pixel_value)**
4. `ToTensorV2` - Converts image and mask to torch.Tensor. The numpy HWC image is converted to pytorch CHW tensor. If the image is in HW format (grayscale image), it will be converted to pytorch HW tensor.
5. `CoarseDropout` - CoarseDropout of the rectangular regions in the image.
```bash
trainTransform = A.Compose([
        A.HorizontalFlip(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=30, interpolation=cv2.INTER_LINEAR, 
                            border_mode=cv2.BORDER_REFLECT_101, always_apply=False, p=0.5),
        A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,
                        fill_value=DATA_MEAN, mask_fill_value = None),
        A.Normalize(DATA_MEAN, DATA_STD),
        ToTensorV2(),
    ])
```
## Model Test accuracy and loss:

Model was trained for 60 epochs.<br>
`Best accuracy` - 89.31%<br>
`Final accuracy` - 88.49%<br>
Model was crossing **accuracy 87%** at epoch 40.<br>
Model [Training Log](./traininglog.md) is attached can be found here<br>

<p align="center">
  <img src="./images/loss.png" width="200" title="accuracy plot">
</p>
    
## Misclassification:

<p align="center">
  <img src="./images/misclassification.png" width="200" title="misclassification">
</p>
    
## Accuracy by Class:

```
Accuracy of plane : 96 %
Accuracy of   car : 96 %
Accuracy of  bird : 70 %
Accuracy of   cat : 77 %
Accuracy of  deer : 94 %
Accuracy of   dog : 79 %
Accuracy of  frog : 98 %
Accuracy of horse : 88 %
Accuracy of  ship : 94 %
Accuracy of truck : 96 %
```

[![ForTheBadge winter-is-coming](http://ForTheBadge.com/images/badges/winter-is-coming.svg)](http://ForTheBadge.com)
    