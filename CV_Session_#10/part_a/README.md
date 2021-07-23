<br/>
<h1 align="center">Session 10: Object Localisation
<br/>
<!-- toc -->
    <br>
    
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
[![Open Source? Yes!](https://badgen.net/badge/Open%20Source%20%3F/Yes%21/blue?icon=github)](https://github.com/RajamannarAanjaram/badges/)
[![Awesome Badges](https://img.shields.io/badge/badges-awesome-green.svg)](https://github.com/RajamannarAanjaram/badges)
    <br>
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![ForTheBadge built-with-love](http://ForTheBadge.com/images/badges/built-with-love.svg)](https://GitHub.com/RajamannarAanjaram/)

### Contributors

<p align="center"> <b>Team - 6</b> <p>
    
| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->
    
## Problem Statement

### Assignment - A

- Download this  TINY IMAGENET dataset. 
- Train ResNet18 on this dataset (70/30 split) for 50 Epochs. Target 50%+ Validation Accuracy. 
- Submit Results. Of course, you are using your own package for everything. You can look at  this  for reference. 

## Model Architecture

  ```
  ----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 64, 64, 64]           1,728
       BatchNorm2d-2           [-1, 64, 64, 64]             128
            Conv2d-3           [-1, 64, 64, 64]          36,864
       BatchNorm2d-4           [-1, 64, 64, 64]             128
            Conv2d-5           [-1, 64, 64, 64]          36,864
       BatchNorm2d-6           [-1, 64, 64, 64]             128
        BasicBlock-7           [-1, 64, 64, 64]               0
            Conv2d-8           [-1, 64, 64, 64]          36,864
       BatchNorm2d-9           [-1, 64, 64, 64]             128
           Conv2d-10           [-1, 64, 64, 64]          36,864
      BatchNorm2d-11           [-1, 64, 64, 64]             128
       BasicBlock-12           [-1, 64, 64, 64]               0
           Conv2d-13          [-1, 128, 32, 32]          73,728
      BatchNorm2d-14          [-1, 128, 32, 32]             256
           Conv2d-15          [-1, 128, 32, 32]         147,456
      BatchNorm2d-16          [-1, 128, 32, 32]             256
           Conv2d-17          [-1, 128, 32, 32]           8,192
      BatchNorm2d-18          [-1, 128, 32, 32]             256
       BasicBlock-19          [-1, 128, 32, 32]               0
           Conv2d-20          [-1, 128, 32, 32]         147,456
      BatchNorm2d-21          [-1, 128, 32, 32]             256
           Conv2d-22          [-1, 128, 32, 32]         147,456
      BatchNorm2d-23          [-1, 128, 32, 32]             256
       BasicBlock-24          [-1, 128, 32, 32]               0
           Conv2d-25          [-1, 256, 16, 16]         294,912
      BatchNorm2d-26          [-1, 256, 16, 16]             512
           Conv2d-27          [-1, 256, 16, 16]         589,824
      BatchNorm2d-28          [-1, 256, 16, 16]             512
           Conv2d-29          [-1, 256, 16, 16]          32,768
      BatchNorm2d-30          [-1, 256, 16, 16]             512
       BasicBlock-31          [-1, 256, 16, 16]               0
           Conv2d-32          [-1, 256, 16, 16]         589,824
      BatchNorm2d-33          [-1, 256, 16, 16]             512
           Conv2d-34          [-1, 256, 16, 16]         589,824
      BatchNorm2d-35          [-1, 256, 16, 16]             512
       BasicBlock-36          [-1, 256, 16, 16]               0
           Conv2d-37            [-1, 512, 8, 8]       1,179,648
      BatchNorm2d-38            [-1, 512, 8, 8]           1,024
           Conv2d-39            [-1, 512, 8, 8]       2,359,296
      BatchNorm2d-40            [-1, 512, 8, 8]           1,024
           Conv2d-41            [-1, 512, 8, 8]         131,072
      BatchNorm2d-42            [-1, 512, 8, 8]           1,024
       BasicBlock-43            [-1, 512, 8, 8]               0
           Conv2d-44            [-1, 512, 8, 8]       2,359,296
      BatchNorm2d-45            [-1, 512, 8, 8]           1,024
           Conv2d-46            [-1, 512, 8, 8]       2,359,296
      BatchNorm2d-47            [-1, 512, 8, 8]           1,024
       BasicBlock-48            [-1, 512, 8, 8]               0
AdaptiveAvgPool2d-49            [-1, 512, 1, 1]               0
           Linear-50                  [-1, 200]         102,600
================================================================
Total params: 11,271,432
Trainable params: 11,271,432
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.05
Forward/backward pass size (MB): 45.01
Params size (MB): 43.00
Estimated Total Size (MB): 88.05
----------------------------------------------------------------
  ```
## Model training log

```
  0%|          | 0/782 [00:00<?, ?it/s]
Epoch 1:
Loss=4.791640281677246 Batch_id=781 Accuracy=4.32: 100%|██████████| 782/782 [01:35<00:00,  8.23it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 2:
Loss=3.7697792053222656 Batch_id=781 Accuracy=9.60: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 3:
Loss=4.1085734367370605 Batch_id=781 Accuracy=13.68: 100%|██████████| 782/782 [01:36<00:00,  8.08it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 4:
Loss=3.6955928802490234 Batch_id=781 Accuracy=16.87: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 5:
Loss=3.7882609367370605 Batch_id=781 Accuracy=19.46: 100%|██████████| 782/782 [01:35<00:00,  8.20it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 6:
Loss=3.2701330184936523 Batch_id=781 Accuracy=22.02: 100%|██████████| 782/782 [01:35<00:00,  8.20it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 7:
Loss=3.6825156211853027 Batch_id=781 Accuracy=23.87: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 8:
Loss=3.2146620750427246 Batch_id=781 Accuracy=25.65: 100%|██████████| 782/782 [01:35<00:00,  8.21it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 9:
Loss=3.3262734413146973 Batch_id=781 Accuracy=27.33: 100%|██████████| 782/782 [01:35<00:00,  8.20it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 10:
Loss=3.4270644187927246 Batch_id=781 Accuracy=29.02: 100%|██████████| 782/782 [01:35<00:00,  8.21it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 11:
Loss=3.142592191696167 Batch_id=781 Accuracy=30.17: 100%|██████████| 782/782 [01:35<00:00,  8.22it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 12:
Loss=2.6232500076293945 Batch_id=781 Accuracy=31.30: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 13:
Loss=3.1652493476867676 Batch_id=781 Accuracy=32.59: 100%|██████████| 782/782 [01:35<00:00,  8.20it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 14:
Loss=2.758798599243164 Batch_id=781 Accuracy=33.66: 100%|██████████| 782/782 [01:35<00:00,  8.20it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 15:
Loss=3.1397650241851807 Batch_id=781 Accuracy=34.72: 100%|██████████| 782/782 [01:36<00:00,  8.10it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 16:
Loss=3.417494773864746 Batch_id=781 Accuracy=35.66: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 17:
Loss=3.3322737216949463 Batch_id=781 Accuracy=36.71: 100%|██████████| 782/782 [01:36<00:00,  8.09it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 18:
Loss=2.9679696559906006 Batch_id=781 Accuracy=37.40: 100%|██████████| 782/782 [01:35<00:00,  8.21it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 19:
Loss=2.251391887664795 Batch_id=781 Accuracy=38.15: 100%|██████████| 782/782 [01:35<00:00,  8.16it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 20:
Loss=2.4217350482940674 Batch_id=781 Accuracy=39.19: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 21:
Loss=3.449260711669922 Batch_id=781 Accuracy=39.95: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 22:
Loss=2.1522395610809326 Batch_id=781 Accuracy=40.79: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 23:
Loss=2.245272159576416 Batch_id=781 Accuracy=41.17: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 24:
Loss=2.279550552368164 Batch_id=781 Accuracy=42.13: 100%|██████████| 782/782 [01:35<00:00,  8.18it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 25:
Loss=2.8508543968200684 Batch_id=781 Accuracy=42.93: 100%|██████████| 782/782 [01:35<00:00,  8.17it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 26:
Loss=2.558565378189087 Batch_id=781 Accuracy=43.40: 100%|██████████| 782/782 [01:35<00:00,  8.21it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 27:
Loss=2.688124895095825 Batch_id=781 Accuracy=44.16: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 28:
Loss=2.611909866333008 Batch_id=781 Accuracy=44.60: 100%|██████████| 782/782 [01:36<00:00,  8.10it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 29:
Loss=2.7099084854125977 Batch_id=781 Accuracy=45.16: 100%|██████████| 782/782 [01:36<00:00,  8.13it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 30:
Loss=2.707061290740967 Batch_id=781 Accuracy=45.70: 100%|██████████| 782/782 [01:35<00:00,  8.19it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 31:
Loss=2.866732358932495 Batch_id=781 Accuracy=46.51: 100%|██████████| 782/782 [01:36<00:00,  8.14it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 32:
Loss=2.3848257064819336 Batch_id=781 Accuracy=46.83: 100%|██████████| 782/782 [01:35<00:00,  8.15it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 33:
Loss=2.3992748260498047 Batch_id=781 Accuracy=47.60: 100%|██████████| 782/782 [01:35<00:00,  8.15it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 34:
Loss=1.975061297416687 Batch_id=781 Accuracy=47.94: 100%|██████████| 782/782 [01:38<00:00,  7.95it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 35:
Loss=2.666930675506592 Batch_id=781 Accuracy=48.52: 100%|██████████| 782/782 [01:35<00:00,  8.18it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 36:
Loss=2.2773711681365967 Batch_id=781 Accuracy=49.17: 100%|██████████| 782/782 [01:35<00:00,  8.16it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 37:
Loss=2.180499792098999 Batch_id=781 Accuracy=49.38: 100%|██████████| 782/782 [01:35<00:00,  8.18it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 38:
Loss=2.280543327331543 Batch_id=781 Accuracy=49.94: 100%|██████████| 782/782 [01:36<00:00,  8.12it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 39:
Loss=2.213228225708008 Batch_id=781 Accuracy=50.28: 100%|██████████| 782/782 [01:35<00:00,  8.16it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 40:
Loss=2.4598333835601807 Batch_id=781 Accuracy=50.63: 100%|██████████| 782/782 [01:35<00:00,  8.15it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 41:
Loss=2.0917835235595703 Batch_id=781 Accuracy=51.35: 100%|██████████| 782/782 [01:35<00:00,  8.17it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 42:
Loss=2.104494094848633 Batch_id=781 Accuracy=51.65: 100%|██████████| 782/782 [01:36<00:00,  8.12it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 43:
Loss=2.0186305046081543 Batch_id=781 Accuracy=52.13: 100%|██████████| 782/782 [01:36<00:00,  8.12it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 44:
Loss=2.2770321369171143 Batch_id=781 Accuracy=52.71: 100%|██████████| 782/782 [01:36<00:00,  8.12it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 45:
Loss=2.1452207565307617 Batch_id=781 Accuracy=52.95: 100%|██████████| 782/782 [01:35<00:00,  8.17it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 46:
Loss=2.0779409408569336 Batch_id=781 Accuracy=53.37: 100%|██████████| 782/782 [01:35<00:00,  8.17it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 47:
Loss=2.5349180698394775 Batch_id=781 Accuracy=53.90: 100%|██████████| 782/782 [01:35<00:00,  8.16it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 48:
Loss=1.735898494720459 Batch_id=781 Accuracy=54.25: 100%|██████████| 782/782 [01:35<00:00,  8.16it/s] 
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 49:
Loss=1.8658931255340576 Batch_id=781 Accuracy=54.40: 100%|██████████| 782/782 [01:35<00:00,  8.17it/s]
  0%|          | 0/782 [00:00<?, ?it/s]


Epoch 50:
Loss=1.8779017925262451 Batch_id=781 Accuracy=54.85: 100%|██████████| 782/782 [01:35<00:00,  8.17it/s]


  ```