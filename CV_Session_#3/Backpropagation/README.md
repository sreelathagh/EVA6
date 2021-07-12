<br/>
<h1 align="center">Session 4: Backpropagation</h1>
<br/>

Table of contents
=================

* [Backpropagation](#BACKPROPAGATION)
   1. [Model Initialisation](#MODEL-INITIALISATION)
   2. [Gradient Calculation](#GRADIENT-CALCULATION)
      * [Forward Pass](#forward-pass)
      * [Backward Pass](#backward-pass)
   4. [Inference](#INFERENCE)


<!-- Backpropagation -->
## Backpropagation

Backpropagation is widely used algorithm for training **`feedforward neural network`**. Backrpopagation computes the gradient in weight space of feedforward 
neural network, with respect to ` loss function`. Some of commonly used loss funciton are **Cross entropy**, **nll loss**, **mse**, etc. The goal of backpropagation
is to optimize the weights so that the neural network can learn how to correctly map inputs to outputs.

<p align="center">
    <img src="./images/simple_perceptron_model.png" alt="Backpropagation" style="zoom: 30%;">
</p>

   <!--Model Initialisation -->

* #### Model Initialisation
    The above image is used to perform backpropagation in excel sheet. It has two input neurons *`i1`* and *`i2`*, these inputs are connected to two hidden 
    neurons *`h1`* and *`h2`*. These hidden neuron is activated using *`sigmoid activation function`*. Then its connnected to output neuron *`o1`* and *`o2`*. 
    Sigmoid activtion is appiled on the ouptu neuron.

        InputParameters:
        i1 = 0.05; i2=0.1
        w1=0.3; w2=0.5, w3=-0.2, w4=0.7
        w5=0.1, w6=-0.6, w7=0.3, w8=-0.9

    <p align="center">
       <img src="./images/modelbuild.png" alt="Backpropagation" style="zoom: 70%;">
    </p>
    <p align="center">
       <sub>sample</sub>
    </p>

* #### Gradient Calculation

   * #### Forward Pass
      We will first pass the above inputs through the network by multiplying the inputs to the weights and calculate the h1 and h2
      ```   
      h1 =w1*i1+w2+i2
      h2 =w3*i1+w4*i2
      ```  
      The output from the hidden layer neurons are passed to activated neurons using a activation function, this helps in adding non linearity to the network.
      ```
      a_h1 = σ(h1) = 1/(1+exp(-h1))
      a_h2 = σ(h2) = 1/(1+exp(-h2))
      
      Here we are using Sigmoid as activation function. `σ`
      
   We repeat this same process for the output layer neurons, we use the activated outputs as input to the output neurons.
   ```
      o1 = w5 * a_h1 + w6 * a_h2
      o2 = w7 * a_h1 + w8 * a_h2
      
      a_o1 = σ(o1) = 1/(1+exp(-o1))
      a_o2 = σ(o2) = 1/(1+exp(-o2))
   ```
   Next, we calculate the error for each output neurons and sum them up to get the total error (E_total). The error we have used here is **`square error function`**.
   ```
      E1 = ½ * ( t1 - a_o1)²
      E2 = ½ * ( t2 - a_o2)²
      E_Total = E1 + E2
   ```
   Note:  1/2 is included so that when calculating gradient doesn't need a constant out front.

   * #### Backward Pass

   The goal of backpropagation is to optimize the weights so that the neural network can learn how to correctly map inputs to outputs. This is done by make the model learn weights so that it will have low error.

   The weight updation starts from output to input. So the weight connected between the hidden layer and output layer gets updated first. The weights connected between these layers are `w5, w6, w7, w8`.<br/>

   First we calculate the partial derivative of E_total with respect to w5 
   ```
      δE_total/δw5 = δ(E1 +E2)/δw5
      
      δE_total/δw5 = δ(E1)/δw5       # removing E2 as there is no impact from E2 wrt w5	
                  = (δE1/δa_o1) * (δa_o1/δo1) * (δo1/δw5)	# Using Chain Rule
                  = (δ(½ * ( t1 - a_o1)²) /δa_o1= (t1 - a_o1) * (-1) = (a_o1 - t1))   # calculate how much does the output of a_o1 change with respect Error
                     * (δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1                       # calculate how much does the output of o1 change with respect a_o1
                     * (1 - a_o1 )) * a_h1                                            # calculate how much does the output of w5 change with respect o1
                  = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
   ```

   Similarly, we calculate the partial derivative of E_total with respect to `w6, w7, w8`.
   ```
      δE_total/δw5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
      δE_total/δw6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
      δE_total/δw7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
      δE_total/δw8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2
   ```
   Now, we calculate the gradients of weights connected between the hidden layer and input neuron w.r.t `E_total`.
   ```
      δE_total/δa_h1 = δ(E1+E2)/δa_h1 
                     = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7
                     
      δE_total/δa_h2 = δ(E1+E2)/δa_h2 
                     = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8
   ```                   
   Calculate the gradient of E_total with respect to w1, w2, w3 and w4 using chain rule   
   ```
      δE_total/δw1 = δE_total/δw1 = δ(E_total)/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1
                  = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
                  
      
      δE_total/δw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2
      δE_total/δw3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1
      δE_total/δw4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2
   ```

   Once we have gradients for all the weights we update them by using the formula

   ```

      weight = weight - learning_rate * gradient_weight_calulated

   ```

> We repeat this entire process for forward and backward pass until we get minimum error.

* #### Inference


The error graph for learning weight - 0.1, 0.2, 0.5, 0.8, 1, 2.

<p align="center">
   <img src="./images/lossoutput1.png" alt="Backpropagation" style="zoom: 50%;">
</p>


    