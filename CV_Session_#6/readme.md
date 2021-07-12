# Session 3 - PyTorch

<!-- toc -->
#### TEAM - 6

| <centre>Name</centre> | <centre>Mail id</centre> | 
| ------------ | ------------- |
| <centre>Amit Agarwal</centre>         | <centre>amit.pinaki@gmail.com</centre>    |
| <centre>Pranav Panday</centre>         | <centre>pranavpandey2511@gmail.com</centre>    |
| <centre>Rajamannar A K</centre>         | <centre>rajamannaraanjaram@gmail.com</centre>    |
| <centre>Sree Latha Chopparapu</centre>         | <centre>sreelathaemail@gmail.com</centre>    |\\

<!-- toc -->

## 1. What is your code all about:
- We have considered the best model from our assignment 5. 
- This code is modularized and programmed to train 3 different models with Group normalization, Layer normalization and Batch normalization with L1 Regularization. 
#### code structure
```
In the folder ->
-- main.ipynb : it’s the driver notebook that invokes all the individual components of the project contained it the src folder. It invokes loader functions to download mnist data and create data loaders. It also intialiases models based on different normalisation parameter, training them one at time &visualization the errors too.
-- src Folder : The folder consists of modularized code snippet files.
-- src/train.py : Consists of training algorithm for GN, LN and BN+L1 regularization.
-- src/test.py : Consists of testing algorithm.
-- src/plots.py : Responsible for plotting the MNIST data samples.
-- src/optimise.py : created a learner with SGD optimization with an lr = 0.015.
-- src/model.py : The Model to train with different normalization techniques.
-- src/data_loader.py: To download the MNIST data.

```
## 3. Our Findings on Normalization:
- 1. Group Normalization: 
        While Training the model by applying group norm, we have experimented with different number of groups. 
        The Key observation is that larger the group, better the performance.
        Training with Smaller groups have a compromise on the accuracy and convergence.
- 2. Layer Normalization: 
        While experimenting with Layer norm its observed that the number of params increases drastically at each layer.
        The accuracy in this case is not as expected. 
- 3. Batch norm + L1 Reg: 
        Batch norm is a good normalization technique for accuracy and faster convergence. 
        But when combined with L1 Reg the performance of the Model is quite dependent on lambda value, higher the lambda value Lower the performance, lower the lambda value for a better performance.

## 3. Model Test accuracy and loss:
<p align="center">
  <img src="./images/loss.png" width="1000" title="hover text">
</p>
