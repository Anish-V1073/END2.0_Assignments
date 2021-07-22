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

    print(embedded_input.shape)
    embedded_input = embedding(input_tensor[0].view(-1, 1))
    print(embedded_input.shape)

Let's build our LSTM, initialize the hidden state and cell state with Zeros(Empty state)

    (hidden,ct) = torch.zeros(1, 1, 256, device=device),torch.zeros(1, 1, 256, device=device)
    embedded_input = embedding(input_tensor[0].view(-1, 1))
    output, (hidden,ct) = lstm(embedded_input, (hidden,ct))
    
Now we will define a empty tensor with size MAX_LENGTH to store the Encoder outputs.<br>
Then we can get the encoder outputs for each of the word in the Sentence

    encoder_outputs = torch.zeros(MAX_LENGTH, 256, device=device)
    (encoder_hidden,encoder_ct) = torch.zeros(1, 1, 256, device=device),torch.zeros(1, 1, 256, device=device)
    
    for i in range(input_tensor.size()[0]):  
      embedded_input = embedding(input_tensor[i].view(-1, 1))
      output, (encoder_hidden,encoder_ct) = lstm(embedded_input, (encoder_hidden,encoder_ct))
      encoder_outputs[i] += output[0,0]

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

![08](https://user-images.githubusercontent.com/62289867/126675318-bb8af8a2-722c-4c8a-8ab2-f0428851b07d.png)
![09](https://user-images.githubusercontent.com/62289867/126673907-04ca02fd-9073-410f-b042-332462e37e1a.png)

     Step 1
     Word => etes
     Input Tensor => tensor(214, device='cuda:0')
![10](https://user-images.githubusercontent.com/62289867/126675476-1b2ccb9f-6666-422c-b598-85dfa4a79c35.png
![11](https://user-images.githubusercontent.com/62289867/126675505-5b473655-a1ae-47f9-bf0b-d04b2e8b4335.png)

    Step 2
    Word => plus
    Input Tensor => tensor(152, device='cuda:0')
![12](https://user-images.githubusercontent.com/62289867/126675659-1f530872-0257-4106-8936-f475e2727f93.png)
![13](https://user-images.githubusercontent.com/62289867/126675669-592b4f2a-e394-4055-bfc2-f4c7ba2a139a.png)

    Step 3
    Word => intelligent
    Input Tensor => tensor(135, device='cuda:0')
![14](https://user-images.githubusercontent.com/62289867/126675766-3540e520-ca24-4206-aa79-8f782cb5aaf5.png)
![15](https://user-images.githubusercontent.com/62289867/126675768-3762222b-b9cf-434a-88bd-5b7ca32f9dc3.png)

    Step 4
    Word => que
    Input Tensor => tensor(902, device='cuda:0')
![16](https://user-images.githubusercontent.com/62289867/126675872-a625f689-68bb-4e61-bb88-03b18efc4199.png)
![17](https://user-images.githubusercontent.com/62289867/126675877-6a51e012-afd7-4043-a045-42067bb3ef49.png)

    Step 5
    Word => moi
    Input Tensor => tensor(42, device='cuda:0')
![18](https://user-images.githubusercontent.com/62289867/126675970-c895b74d-970e-4432-851b-557826599f87.png)
![19](https://user-images.githubusercontent.com/62289867/126675983-fcff1815-c2e9-4c4b-8998-6d63b1608ca5.png)

    Step 6
    Word => .
    Input Tensor => tensor(5, device='cuda:0')
![20](https://user-images.githubusercontent.com/62289867/126676176-99cb5fae-0d71-470e-ac8f-cfa401afa71f.png)
![21](https://user-images.githubusercontent.com/62289867/126676184-9761d3f8-ae23-4c8a-b0ad-1533ddfe73d0.png)

    Step 7
    Word => <EOS>
    Input Tensor => tensor(1, device='cuda:0')
![22](https://user-images.githubusercontent.com/62289867/126676328-af7a7972-6b6b-459a-8a66-e1eb34cae0ee.png)
![23](https://user-images.githubusercontent.com/62289867/126676345-73a30374-fc6e-40ea-98bb-f34a72390791.png)


We completed the Encoder part now, Now we can start building the Attention Decoder<br>
* First input to the decoder will be SOS_token, later inputs would be the words it predicted (unless we implement teacher forcing).
* Decoder/LSTM's hidden state will be initialized with the encoder's last hidden state.
* We will use LSTM's hidden state and last prediction to generate attention weight using a FC layer.
* This attention weight will be used to weigh the encoder_outputs using batch matric multiplication. This will give us a NEW view on how to look at encoder_states.
* this attention applied encoder_states will then be concatenated with the input, and then sent a linear layer and then sent to the LSTM.
* LSTM's output will be sent to a FC layer to predict one of the output_language words


      # first input
      decoder_input = torch.tensor([[SOS_token]], device=device)
      (decoder_hidden,decoder_ct) = (encoder_hidden,encoder_ct)
      decoded_words = []
      
 We need to concatenate the embeddings and the last decoder hidden state
 
     torch.cat((embedded[0], decoder_hidden[0]), 1).shape
     
     => torch.Size([1, 512])
     
Now we will calaculate the attentions. We will calculating the attentions by conacatinating the embeddings and last decoder hidden state and giving as input to the fully connected layer.

    attn_weight_layer = nn.Linear(256 * 2, 10).to(device)
    attn_weights = attn_weight_layer(torch.cat((embedded[0], decoder_hidden[0]), 1))
    attn_weights
    
    =>tensor([[-0.8181,  0.0128,  0.0196, -0.3952, -0.1043, -0.1855, -0.5074, -0.4552,
         -0.5731,  0.5895]], device='cuda:0', grad_fn=<AddmmBackward>)
         
### Decoder steps

    Step 0
    Expected output(word) => you 
    Expected output(Index) => 129 
    Predicted output(word) => opposed 
    Predicted output(Index) => 2669 
![24](https://user-images.githubusercontent.com/62289867/126677215-73f6ed09-9fed-4c4b-b1f3-dd41f0590cac.png)

    Step 1
    Expected output(word) => re 
    Expected output(Index) => 78 
    Predicted output(word) => opposed 
    Predicted output(Index) => 2669 
![25](https://user-images.githubusercontent.com/62289867/126677286-80ea1d6e-38c3-4db4-91ec-43b1826bec87.png)

    Step 2
    Expected output(word) => smarter 
    Expected output(Index) => 1319 
    Predicted output(word) => opposed 
    Predicted output(Index) => 2669 
![26](https://user-images.githubusercontent.com/62289867/126677335-6f9c1a1c-3f1d-4d24-9474-fff9cfd0071b.png)


    Step 3
    Expected output(word) => than 
    Expected output(Index) => 1166 
    Predicted output(word) => options 
    Predicted output(Index) => 1343 
![27](https://user-images.githubusercontent.com/62289867/126677438-062ce8ec-08f8-4a62-88c9-fd70d02ba11a.png)


    Step 4
    Expected output(word) => me 
    Expected output(Index) => 343 
    Predicted output(word) => articulate 
    Predicted output(Index) => 964 
![28](https://user-images.githubusercontent.com/62289867/126677542-0fda28b7-ae24-4cd2-a42d-ee14c3e100fb.png)


    Step 5
    Expected output(word) => . 
    Expected output(Index) => 4 
    Predicted output(word) => forgetting 
    Predicted output(Index) => 2345 
![29](https://user-images.githubusercontent.com/62289867/126677640-18cdfb6b-6c7f-450e-9bd4-4b4e8ffca48c.png)






