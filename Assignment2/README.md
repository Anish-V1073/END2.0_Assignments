## Scope

Finding back propagation for the below neural network.

<img width="542" alt="Screenshot 2021-05-11 163547" src="https://user-images.githubusercontent.com/62289867/117949475-c8491680-b32f-11eb-8b20-ffa4640bf32b.png">

### Worksheet of the above neural network

[Assignment2.xlsx](https://github.com/Anish-V1073/END2.0_Assignments/files/6464976/Assignment2.xlsx)


### Screenshot of calculation table with learning rate 0.5
<img width="926" alt="Screenshot 2021-05-11 194744" src="https://user-images.githubusercontent.com/62289867/117950354-b87e0200-b330-11eb-9f3c-1902fc6e8389.png">


### Error graph with different learning rates
#### Learning rate 0.1
<img width="369" alt="Screenshot 2021-05-11 195059" src="https://user-images.githubusercontent.com/62289867/117950850-3b06c180-b331-11eb-9680-b1eb1cedf63d.png">

#### Learning rate 0.2
<img width="370" alt="Screenshot 2021-05-11 195150" src="https://user-images.githubusercontent.com/62289867/117950941-4f4abe80-b331-11eb-9b62-d1ec1a8b5fbb.png">

#### Learning rate 0.3
<img width="368" alt="Screenshot 2021-05-11 195229" src="https://user-images.githubusercontent.com/62289867/117951227-9638b400-b331-11eb-80f4-a1bf17097078.png">

#### Learning rate 0.5
<img width="373" alt="Screenshot 2021-05-11 195010" src="https://user-images.githubusercontent.com/62289867/117951286-a81a5700-b331-11eb-80ba-8f3974fdff8f.png">

#### Learning rate 0.8
<img width="370" alt="Screenshot 2021-05-11 195322" src="https://user-images.githubusercontent.com/62289867/117951369-bbc5bd80-b331-11eb-90dd-f299cf5e9cdf.png">

#### Learning rate 1
<img width="369" alt="Screenshot 2021-05-11 195400" src="https://user-images.githubusercontent.com/62289867/117951422-c84a1600-b331-11eb-85fe-cd3409a6a338.png">

#### Learning rate 1.5
<img width="369" alt="Screenshot 2021-05-11 195453" src="https://user-images.githubusercontent.com/62289867/117951493-d9932280-b331-11eb-8042-65df33374c35.png">

#### Learning rate 2
<img width="371" alt="Screenshot 2021-05-11 195551" src="https://user-images.githubusercontent.com/62289867/117951566-ea439880-b331-11eb-9e2d-e364741aa8ff.png">

### Gradient calculation with respect to the weights

The error contribution to the total error w.r.t weight **w5** will computed as below:

    ∂E_tot/∂w5  = ∂(E1+E2)/∂w5 
                = ∂(E1)/∂w5  ( E2 is eliminated because there is no contribution to E2 by w5)
                = ∂(E1)/∂w5 * ∂a_o1/∂o1*∂o/∂w5
                = ∂(1/2*(t1-a_o1)^2)/∂d_o1* σ(o1)/∂o1 * a_h1
                = 1/2 *(2 * (t1-a_o1) *∂(t1-a_o1)/∂a_o1 * a_o1* (1-a_o1) * a_h1
                = (t1 - a_o1)(0-1) * a_o1 * (1-a_o1) * a_h1
    ∂E_tot/∂w5  = (a_o1 - t1) * a_o1 * (1-a_o1) * a_h1
            

The error contribution to the total error w.r.t weight **w6** will computed as above. Hence the final equation will be as below:

    ∂E_tot/∂w6  = (a_o1-t1)*a_o1*(1-a_o1)*a_h2

The error contribution to the total error w.r.t weight **w7** will computed as above. Hence the final equation will be as below:

    ∂E_tot/∂w7 = (a_o2-t2)*a_o2*(1-a_o2)*a_h1

The error contribution to the total error w.r.t weight **w8** will computed as above. Hence the final equation will be as below:

    ∂E_tot/∂w8 = (a_o2-t2)*a_o2*(1-a_o2)*a_h2

The error contribution to the total error w.r.t weight **w1** will computed as above. Hence the final equation will be as below:

    ∂E_tot/dw1    = ∂E_tot/∂a_o1*∂a_o1/∂o1*∂o1/∂a_h1*∂a_h1/∂h1*∂h1/∂w1 = ∂E_tot/∂a_h1*∂a_h1/∂h1*∂h1/∂w1
    ∂E_tot/∂a_h1  = ∂(E1+E2)/∂a_h1
    ∂E1/∂a_h1     = ∂E1/∂a_o1*∂a_o1/∂o1*∂o1/∂a_h1
                  = (a_o1-t1)*a_o1*(1-a_o1)*w5
    ∂E2/∂a_h1     = (a_o2-t2)*a_o2*(1-a_o2)*w7
    ∂E_tot/∂a_h1  = (a_o1-t1)*a_o1*(1-a_o1)*w5 + (a_o2-t2)*a_o2*(1-a_o2)*w7
    ∂E_tot/∂a_h2  = (a_o1-t1)*a_o1*(1-a_o1)*w6 + (a_o2-t2)*a_o2*(1-a_o2)*w8
    ∂E_tot/∂w1    = ∂E_tot/∂a_o1*∂a_o1/∂o1*∂o1/∂a_h1*∂a_h1/∂h1*∂h1/∂w1 
                  = ∂E_tot/∂a_h1*∂a_h1/∂h1*∂h1/∂w1
                  = ∂E_tot/∂a_h1 * a_h1*(1-a_h1)*i1
    ∂E_tot/∂w1    = ((a_o1-t1)*a_o1*(1-a_o1)*w5 + (a_o2-t2)*a_o2*(1-a_o2)*w7)*a_h1*(1-a_h1)*i1   
    
The error contribution to the total error w.r.t weight **w2** will computed as above. Hence the final equation will be as below:

    ∂E_tot/∂w2    = ((a_o1-t1)*a_o1*(1-a_o1)*w5 + (a_o2-t2) * a_h1*(1-a_h1)*i2
    
The error contribution to the total error w.r.t weight **w3** will computed as above. Hence the final equation will be as below:   

    ∂E_tot/∂w3 = ((a_o1-t1)*a_o1*(1-a_o1)*w6 + (a_o2-t2)*a_o2*(1-a_o2)*w8) * a_h2*(1-a_h2)*i1
    
The error contribution to the total error w.r.t weight **w3** will computed as above. Hence the final equation will be as below: 
    
    ∂E_tot/∂w4 = ((a_o1-t1)*a_o1*(1-a_o1)*w6 + (a_o2-t2)*a_o2*(1-a_o2)*w8) * a_h2*(1-a_h2)*i2
    
Formula to update the new weights:

    new_weight = old_weight - LR * Gradient of associated weight
    
    


