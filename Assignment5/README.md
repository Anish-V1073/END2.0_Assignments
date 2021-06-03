# Sentiment Analysis using LSTM 

## Objective
Write a neural network(LSTM) to detect the sentiment of given sentence in Stanford Sentiment Analysis dataset.
* Use "Back Translate", "Random Swap" and "Random Delete" to augment the data used for training.
* Train your model and achieve 60% + validation/test accuracy. 

### About Stanford Sentiment Analysis dataset.
It consists of 10,605 processed snippets from the original pool of Rotten Tomatoes HTML files. The original snippets are parsed using Stanford parser and generated a multiple phrases from each sentence. This dataset consits of phrases,phrase_id and sentiment labels. We can recover the  classes by mapping the positivity probability using the following cut-offs:

[0, 0.2]   --> very negative

[0.2, 0.4] --> negative

[0.4, 0.6] --> neutral

[0.6, 0.8] --> positive

[0.8, 1.0] --> very positive

The dataset also consits of sentences along with the sentiment values for 5 class in PTB(Penn Treebank) format. We have used the Pytreebank python library to parse and obtain the fine grained 5 calss sentiment values from the Penn Treebank files. [Link](https://towardsdatascience.com/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4).

# Proposed solution
We applied a data augmentation techniques like "Back Translate", "Random swap" and "Random delete" on train dataset.

* **Back Translate** - In this method, we translate the text data to some random language and then translate it back to the original language. This can help to generate textual data with different words while preserving the context of the text data. 
* **Random delete** -  In this method, we randomly remove each word in the sentence with probability p.
* **Random swap** - In this method, we randomly choose two words in the sentence and swap their positions. Do this n times. 
   
For example, we will take a sample data from dataset
  
    back_translate("The camera twirls ! Oh , look at that clever angle ! Wow , a jump cut !")
    Result:The camera will help! Oh, look at this smart angle! Wow, jump back!
     
    random_deletion("The camera twirls ! Oh , look at that clever angle ! Wow , a jump cut !")
    Result:The twirls ! Oh , look at that clever angle ! Wow , a jump cut !

    random_swap("The camera twirls ! Oh , look at that clever angle ! Wow , a jump cut !")
    Result:The camera twirls ! Oh , look at that clever jump ! Wow , a angle cut !
    
And also we performed a data cleaning process by removing the punctuations in the sentence.For example, 

    preprocess("The camera twirls ! Oh , look at that clever angle ! Wow , a jump cut !")
    Result: The camera twirls Oh look at that clever angle Wow a jump cut

## The Network

![06](https://user-images.githubusercontent.com/36162708/120683107-da475080-c4ba-11eb-9555-2ddd016125f6.jpg)

The Input text is preprocessed by removing a punctuations, Tokenized with spacy tokenizer and passed to embedding layer.

Embedding Dimensions = 300

then the embedded tokens are passed to a 2 layer LSTM.

* Input size = 100
* Hidden Nodes = 50 
* Output Nodes = 5
* No. of layers = 2
* batch_first = True
* Dropout = 0.3

The output of the LSTM is given to a fully connected linear layer
* in_features : 50
* out_features: 5

Network:

      classifier(
         (embedding): Embedding(24428, 100)
         (encoder): LSTM(100, 50, num_layers=2, batch_first=True, dropout=0.3)
         (fc2): Linear(in_features=50, out_features=5, bias=True)
         (dropout): Dropout(p=0.3, inplace=False)
      )
      The model has 2,493,855 trainable parameters
      
 ### Training logs
 
      Epoch: 0 Train Loss: 1.612 | Train Acc: 17.79% Val. Loss: 1.609 |  Val. Acc: 19.91%
      Epoch: 1 Train Loss: 1.609 | Train Acc: 20.21% Val. Loss: 1.607 | Val. Acc: 23.59%
      Epoch: 2 Train Loss: 1.606 | Train Acc: 23.57% Val. Loss: 1.605 | Val. Acc: 27.51%
      Epoch: 3 Train Loss: 1.602 | Train Acc: 26.42% Val. Loss: 1.601 | Val. Acc: 28.18%
      Epoch: 4 Train Loss: 1.596 | Train Acc: 28.38% Val. Loss: 1.595 | Val. Acc: 28.97%
      Epoch: 5 Train Loss: 1.585 | Train Acc: 28.80% Val. Loss: 1.585 | Val. Acc: 25.95%
      Epoch: 6 Train Loss: 1.575 | Train Acc: 28.59% Val. Loss: 1.582 | Val. Acc: 25.64%
      Epoch: 7 Train Loss: 1.571 | Train Acc: 28.70% Val. Loss: 1.580 | Val. Acc: 26.56%
      Epoch: 8 Train Loss: 1.570 | Train Acc: 29.15% Val. Loss: 1.579 | Val. Acc: 26.81% 
      . 
      . 
      . 
      . 
      . 
      Epoch: 143 Train Loss: 1.199 | Train Acc: 71.68% Val. Loss: 1.556 | Val. Acc: 33.57%
      Epoch: 144 Train Loss: 1.197 | Train Acc: 71.76% Val. Loss: 1.555 | Val. Acc: 33.48%
      Epoch: 145 Train Loss: 1.196 | Train Acc: 71.89% Val. Loss: 1.556 | Val. Acc: 33.44%
      Epoch: 146 Train Loss: 1.195 | Train Acc: 71.93% Val. Loss: 1.556 | Val. Acc: 33.61%
      Epoch: 147 Train Loss: 1.194 | Train Acc: 72.19% Val. Loss: 1.556 | Val. Acc: 33.66%
      Epoch: 148 Train Loss: 1.192 | Train Acc: 72.36% Val. Loss: 1.555 | Val. Acc: 33.70%
      Epoch: 149 Train Loss: 1.191 | Train Acc: 72.46% Val. Loss: 1.557 | Val. Acc: 33.83%

### Results

      Minimum Train Loss: 1.191   
      Minimum Validation Loss: 1.58
   
   
      Maximum Train Accuracy: 72.46 %   
      Maximum Validation Accuracy: 33.83%  
   
   ![02](https://user-images.githubusercontent.com/36162708/120681906-825c1a00-c4b9-11eb-8cff-4090593f7a1e.png)![01](https://user-images.githubusercontent.com/36162708/120681926-85efa100-c4b9-11eb-841d-88a341618a46.png)
   ![03](https://user-images.githubusercontent.com/36162708/120681985-93a52680-c4b9-11eb-82de-9360b54f49c6.png)![04](https://user-images.githubusercontent.com/36162708/120682019-9c95f800-c4b9-11eb-81ce-4474f03de649.png)


   
### Outcomes:

   ## Correct Prediction
   
      Sample Text: The story has some nice twists but the ending and some of the back-story is a little tired
      Actual Value: 2
      Predicted Value: 2
      
      Sample Text: A naturally funny film , Home Movie makes you crave Chris Smith 's next movie
      Actual Value: 5
      Predicted Value: 5
      
      Sample Text: Pipe Dream does have its charms
      Actual Value: 4
      Predicted Value: 4
   
 #### Wrong Prediction
      
      Sample Text: Great over-the-top moviemaking if you 're in a slap-happy mood
      Actual Value: 5
      Predicted Value: 4
      
      Sample Text: A tasty appetizer that leaves you wanting more
      Actual Value: 4
      Predicted Value: 5
      
      Sample Text: The history is fascinating ; the action is dazzling
      Actual Value: 5
      Predicted Value: 4
      
      Sample Text: Igby Goes Down is one of those movies
      Actual Value: 3
      Predicted Value: 2
      
      Sample Text: It turns out to be smarter and more diabolical than you could have guessed at the beginning
      Actual Value: 5
      Predicted Value: 4
      
      Sample Text: In between all the emotional seesawing , it 's hard to figure the depth of these two literary figures , and even the times in which they lived
      Actual Value: 2
      Predicted Value: 3
      
      Sample Text: He watches them as they float within the seas of their personalities
      Actual Value: 4
      Predicted Value: 3

