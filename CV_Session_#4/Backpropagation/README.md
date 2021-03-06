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
      a_h1 = ??(h1) = 1/(1+exp(-h1))
      a_h2 = ??(h2) = 1/(1+exp(-h2))
      
      Here we are using Sigmoid as activation function. `??`
      
   We repeat this same process for the output layer neurons, we use the activated outputs as input to the output neurons.
   ```
      o1 = w5 * a_h1 + w6 * a_h2
      o2 = w7 * a_h1 + w8 * a_h2
      
      a_o1 = ??(o1) = 1/(1+exp(-o1))
      a_o2 = ??(o2) = 1/(1+exp(-o2))
   ```
   Next, we calculate the error for each output neurons and sum them up to get the total error (E_total). The error we have used here is **`square error function`**.
   ```
      E1 = ?? * ( t1 - a_o1)??
      E2 = ?? * ( t2 - a_o2)??
      E_Total = E1 + E2
   ```
   Note:  1/2 is included so that when calculating gradient doesn't need a constant out front.

   * #### Backward Pass

   The goal of backpropagation is to optimize the weights so that the neural network can learn how to correctly map inputs to outputs. This is done by make the model learn weights so that it will have low error.

   The weight updation starts from output to input. So the weight connected between the hidden layer and output layer gets updated first. The weights connected between these layers are `w5, w6, w7, w8`.<br/>

   First we calculate the partial derivative of E_total with respect to w5 
   ```
      ??E_total/??w5 = ??(E1 +E2)/??w5
      
      ??E_total/??w5 = ??(E1)/??w5       # removing E2 as there is no impact from E2 wrt w5	
                  = (??E1/??a_o1) * (??a_o1/??o1) * (??o1/??w5)	# Using Chain Rule
                  = (??(?? * ( t1 - a_o1)??) /??a_o1= (t1 - a_o1) * (-1) = (a_o1 - t1))   # calculate how much does the output of a_o1 change with respect Error
                     * (??(??(o1))/??o1 = ??(o1) * (1-??(o1)) = a_o1                       # calculate how much does the output of o1 change with respect a_o1
                     * (1 - a_o1 )) * a_h1                                            # calculate how much does the output of w5 change with respect o1
                  = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
   ```

   Similarly, we calculate the partial derivative of E_total with respect to `w6, w7, w8`.
   ```
      ??E_total/??w5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
      ??E_total/??w6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
      ??E_total/??w7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
      ??E_total/??w8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2
   ```
   Now, we calculate the gradients of weights connected between the hidden layer and input neuron w.r.t `E_total`.
   ```
      ??E_total/??a_h1 = ??(E1+E2)/??a_h1 
                     = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7
                     
      ??E_total/??a_h2 = ??(E1+E2)/??a_h2 
                     = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8
   ```                   
   Calculate the gradient of E_total with respect to w1, w2, w3 and w4 using chain rule   
   ```
      ??E_total/??w1 = ??E_total/??w1 = ??(E_total)/??a_o1 * ??a_o1/??o1 * ??o1/??a_h1 * ??a_h1/??h1 * ??h1/??w1
                  = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
                  
      
      ??E_total/??w2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2
      ??E_total/??w3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1
      ??E_total/??w4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2
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


    