## Scope

Finding back propagation for the below neural network with the example.

<img width="542" alt="Screenshot 2021-05-11 163547" src="https://user-images.githubusercontent.com/62289867/117949475-c8491680-b32f-11eb-8b20-ffa4640bf32b.png">

### Worksheet of the above neural network

[Assignment2.xlsx](https://github.com/Anish-V1073/END2.0_Assignments/files/6464976/Assignment2.xlsx)




### What is the use of the learning rate?
The learning rate is a configurable hyperparameter used in the training of neural networks that has a small positive value.

The learning rate controls how quickly the model is adapted to the problem. A smaller learning rate requires many updates before reaching the minimum point. Too large of a learning rate causes drastic updates which lead to divergent behaviors. So it is always important to choose optimal learning rate swiftly reaches the minimum point.

### How are weights intialized?
We can initialize the weights for neural network with differnt techinques such as Zero initialization, Random initialization, He initialization, Xavier initialization based on our network. The selected weight initialization technique should not affect the training purpose.

### What is "loss" in neural network?
  "Loss" is nothing but a prediction error. This is the difference between the expected output and predicted output.

### What is "chain rule" in gradient flow?
  The algorithm used to update the model parameters(weights, biases) in order to effectively train a neural network is known as chain rule.  
  Mathematically total output gradient is the total gradient caused by the all the neurons which are contributed for a output:  
    
  FinalGradient = GradientContribution**1** + GradientContribution**2**+ ....+ GradientContribution**N**
   
  Gradient<sub>i</sub> = GradientInside × GradientContribution<sub>i</sub> <!--∂Output∂wi=∂Contribution1∂wi×∂Output∂Contribution1 -->  
   
  <img align="center" src="https://render.githubusercontent.com/render/math?math=\frac{\partial _{Output}}{\partial _{w^i}} = \frac{\partial _{Contribution^i}}{\partial _{w^i}}    \times \frac{\partial _{Output}}{\partial _{Contribution^i}} ">
   
<img width="542" alt="Screenshot 2021-05-11 163547" src="https://user-images.githubusercontent.com/62289867/117949415-bb2c2780-b32f-11eb-9acc-ddd3d908c035.png">
