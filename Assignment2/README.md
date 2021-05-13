## Scope

Finding back propagation for the below neural network.
<p align="center">
<img width="486" alt="Network Image" src="https://user-images.githubusercontent.com/62289867/118082366-afe20600-b3da-11eb-931b-90d5094277b3.png">
</p>


### Worksheet of the above neural network

[Assignment2.xlsx](https://github.com/Anish-V1073/END2.0_Assignments/files/6464976/Assignment2.xlsx)


### Screenshot of calculation table with learning rate 0.5
<p align="center">
<img width="926" alt="Screenshot 2021-05-11 194744" src="https://user-images.githubusercontent.com/62289867/117950354-b87e0200-b330-11eb-9f3c-1902fc6e8389.png">
</p>

### Error graph with different learning rates
|     |     |
|:---:|:---:|
|<img width="369" alt="Screenshot 2021-05-11 195059" src="https://user-images.githubusercontent.com/62289867/117950850-3b06c180-b331-11eb-9680-b1eb1cedf63d.png">|<img width="370" alt="Screenshot 2021-05-11 195150" src="https://user-images.githubusercontent.com/62289867/117950941-4f4abe80-b331-11eb-9b62-d1ec1a8b5fbb.png">|
|<img width="370" alt="Screenshot 2021-05-11 195150" src="https://user-images.githubusercontent.com/62289867/117950941-4f4abe80-b331-11eb-9b62-d1ec1a8b5fbb.png">|<img width="368" alt="Screenshot 2021-05-11 195229" src="https://user-images.githubusercontent.com/62289867/117951227-9638b400-b331-11eb-80f4-a1bf17097078.png">|
|<img width="373" alt="Screenshot 2021-05-11 195010" src="https://user-images.githubusercontent.com/62289867/117951286-a81a5700-b331-11eb-80ba-8f3974fdff8f.png">|<img width="370" alt="Screenshot 2021-05-11 195322" src="https://user-images.githubusercontent.com/62289867/117951369-bbc5bd80-b331-11eb-90dd-f299cf5e9cdf.png">|
|<img width="369" alt="Screenshot 2021-05-11 195400" src="https://user-images.githubusercontent.com/62289867/117951422-c84a1600-b331-11eb-85fe-cd3409a6a338.png">|<img width="369" alt="Screenshot 2021-05-11 195453" src="https://user-images.githubusercontent.com/62289867/117951493-d9932280-b331-11eb-8042-65df33374c35.png">|
|<img width="371" alt="Screenshot 2021-05-11 195551" src="https://user-images.githubusercontent.com/62289867/117951566-ea439880-b331-11eb-9e2d-e364741aa8ff.png">| |

### Gradient calculation with respect to the weights

The error contribution to the total error w.r.t weight **w5** will computed as below:

    ∂E_tot/∂w5  = ∂(E1+E2)/∂w5 
                = ∂(E1)/∂w5  ( E2 is eliminated because there is no contribution to E2 by w5)
                = ∂(E1)/∂w5 * ∂a_o1/∂o1 * ∂o/∂w5
                = ∂(1/2*(t1-a_o1)^2)/∂d_o1 * ∂σ(o1)/∂o1 * a_h1
                
The derivative of σ(x) = x * (1-x), so σ(o1) w.r.t o1 equals a_o1 * (1-a_o1)

                = 1/2 *(2*(t1-a_o1)*∂(t1-a_o1)/∂a_o1 * (a_o1 * (1-a_o1)) * a_h1
                = (t1 - a_o1)(0-1) * a_o1 * (1-a_o1) * a_h1
    ∂E_tot/∂w5  = (a_o1 - t1) * a_o1 * (1-a_o1) * a_h1
            

Similarly, error contribution to the total error w.r.t weight **w6,w7,w8**, will be as below:

    ∂E_tot/∂w6 = (a_o1-t1) * (a_o1*(1-a_o1)) * a_h2
    
    ∂E_tot/∂w7 = (a_o2-t2) * (a_o2*(1-a_o2)) * a_h1
    
    ∂E_tot/∂w8 = (a_o2-t2) * (a_o2*(1-a_o2)) * a_h2
    

The error contribution to the total error w.r.t weight **w1** will computed as below:<br>
The Total error contributed by **w1** is the sum of the error contributed from **[w1 -> E1]** and **[w1 -> E2]**(Refer the path highlighted in orange colour)
<p align="center">
<img width="559" alt="Network Image 1" src="https://user-images.githubusercontent.com/62289867/118122149-103f6a80-b410-11eb-997c-a6bc8a38e857.png">
</p>

    ∂E_tot/dw1    = ∂E_tot/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1 = ∂E_tot/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
    ∂E_tot/∂a_h1  = ∂(E1+E2)/∂a_h1
    ∂E1/∂a_h1     = ∂E1/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1
                  = (a_o1-t1) * a_o1*(1-a_o1) * w5
    ∂E2/∂a_h1     = (a_o2-t2) * a_o2*(1-a_o2) * w7
    ∂E_tot/∂a_h1  = (a_o1-t1) * a_o1*(1-a_o1) *w5 + (a_o2-t2) * a_o2*(1-a_o2) * w7
    ∂E_tot/∂a_h2  = (a_o1-t1) * a_o1*(1-a_o1) *w6 + (a_o2-t2) *a_o2 *(1-a_o2) * w8
    
    ∂E_tot/∂w1    = ∂E_tot/∂a_o1 * ∂a_o1/∂o1 * ∂o1/∂a_h1 * ∂a_h1/∂h1 * ∂h1/∂w1
    
Since we already calculated the error from **[a_h1->o1->a_h1]** and **[a_h1->o2->a_h2]** (i.e ∂E_tot/∂a_h1)  we can rewrite the above equation as,

                  = ∂E_tot/∂a_h1 *∂a_h1/∂h1 * ∂h1/∂w1
                  = ∂E_tot/∂a_h1 * a_h1*(1-a_h1) * i1
    ∂E_tot/∂w1    = ((a_o1-t1) * a_o1*(1-a_o1) * w5 + (a_o2-t2) * a_o2*(1-a_o2) * w7)* a_h1*(1-a_h1) * i1   
    
Similarly, error contribution to the total error w.r.t weight **w2,w3,w4** will be as below:

    ∂E_tot/∂w2 = ((a_o1-t1) * a_o1*(1-a_o1) * w5 + (a_o2-t2) * a_o2*(1-a_o2) * w7) * a_h1*(1-a_h1) * i2
    
    ∂E_tot/∂w3 = ((a_o1-t1) * a_o1*(1-a_o1) * w6 + (a_o2-t2) * a_o2*(1-a_o2) * w8) * a_h2*(1-a_h2) * i1    

    ∂E_tot/∂w4 = ((a_o1-t1) * a_o1*(1-a_o1) * w6 + (a_o2-t2) * a_o2*(1-a_o2) * w8) * a_h2*(1-a_h2) * i2
    
Formula to update the new weights:

    new_weight = old_weight - LR * Gradient of associated weight
    
    


