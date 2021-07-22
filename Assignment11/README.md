# Assignment 11

## Objective
* Follow the similar strategy as we did in our [baby-steps-code](https://colab.research.google.com/drive/1IlorkvXhZgmd_sayOVx4bC_I5Qpdzxk_?usp=sharing), but replace GRU with LSTM. In your code you must:
  * Perform 1 full feed forward step for the encoder manually
  * Perform 1 full feed forward step for the decoder manually.
  * You can use any of the 3 attention mechanisms that we discussed. 
* Explain your steps in the readme file and
* Submit the assignment asking for these things:
  * Link to the readme file that must explain Encoder/Decoder Feed-forward manual steps and the attention mechanism that you have used - 500 pts
  * Copy-paste (don't redirect to github), the Encoder Feed Forward steps for 2 words - 250 pts
  * Copy-paste (don't redirect to github), the Decoder Feed Forward steps for 2 words - 250 pts
 
 
### Solution

 Select a random pair of senetences from the data we prepared
 
    sample = random.choice(pairs)
    sample
    
    =>['vous etes plus intelligent que moi .', 'you re smarter than me .']   

In order to work with embedding layer and the LSTM the inputs should be in the form of tensor, So we need to convert the sentences(words) to tensors.<br>
First we'll split the sentences by whitespaces and convert each words into indices(using word2index[word])

    input_sentence = sample[0]
    target_sentence = sample[1]
    input_indices = [input_lang.word2index[word] for word in input_sentence.split(' ')]
    target_indices = [output_lang.word2index[word] for word in target_sentence.split(' ')]
    input_indices, target_indices
    
    =>([118, 214, 152, 135, 902, 42, 5], [129, 78, 1319, 1166, 343, 4])   
    
Then convert the input_indices into tensors

    input_tensor = torch.tensor(input_indices, dtype=torch.long, device= device)
    output_tensor = torch.tensor(target_indices, dtype=torch.long, device= device)

Next, We will define a Embedding layer as well as LSTM layers for encoder

    embedding = nn.Embedding(input_size, hidden_size).to(device)
    lstm = nn.LSTM(hidden_size, hidden_size).to(device)

We are working with 1 sample, but we would be working for a batch. Let's fix that by converting our input_tensor into a fake batch
![05](https://user-images.githubusercontent.com/62289867/126673047-a0394e14-feb5-4414-9220-188388e57e28.png)

Let's build our LSTM, initialize the hidden state and cell state with Zeros(Empty state)
![06](https://user-images.githubusercontent.com/62289867/126673298-44079ece-5a32-45af-b10a-e347b219adf9.png)

Now we will define a empty tensor with size MAX_LENGTH to store the Encoder outputs.<br>
Then we can get the encoder outputs for each of the word in the Sentence
![07](https://user-images.githubusercontent.com/62289867/126673467-0ad73171-c097-4ccc-b82b-59b52e5389b0.png)

### Encoder Steps

    Input Sentence: vous etes plus intelligent que moi .
    Target Sentence: you re smarter than me .
    Input indices: [118, 214, 152, 135, 902, 42, 5]
    Target indices: [118, 214, 152, 135, 902, 42, 5]
    After adding the <EOS> token
    Input indices: [118, 214, 152, 135, 902, 42, 5, 1]
    Target indices: [118, 214, 152, 135, 902, 42, 5, 1]
    Input tensor: tensor([118, 214, 152, 135, 902,  42,   5,   1], device='cuda:0')
    Target tensor: tensor([ 129,   78, 1319, 1166,  343,    4,    1], device='cuda:0')
    
    Step 0
    Word => vous

    Input Tensor => tensor(118, device='cuda:0')

![08](https://user-images.githubusercontent.com/62289867/126674300-11f055cd-0a70-4db3-a5be-e9b8b3c68135.png)

![09](https://user-images.githubusercontent.com/62289867/126673907-04ca02fd-9073-410f-b042-332462e37e1a.png)





