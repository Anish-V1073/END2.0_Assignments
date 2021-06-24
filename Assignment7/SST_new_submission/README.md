# Stanford Sentiment Analysis dataset prepration

## Objective
* Prepare a dataset from SST databank without using any libraries(to parse the SST Tree), split dataset into 70/30 Train and Test (no validation) and Convert floating-point labels into 5 classes (0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8, 0.8-1.0) 

### About Stanford Sentiment Analysis dataset.
It consists of 10,605 processed snippets from the original pool of Rotten Tomatoes HTML files. The original snippets are parsed using Stanford parser and generated a multiple phrases from each sentence. This dataset consits of phrases,phrase_id and sentiment labels. We can recover the  classes by mapping the positivity probability using the following cut-offs:

[0, 0.2]   --> very negative

[0.2, 0.4] --> negative

[0.4, 0.6] --> neutral

[0.6, 0.8] --> positive

[0.8, 1.0] --> very positive

### Dataset Prepration

The "datasetSentences.txt" file in the SST databank consits of sentences along with their sentence_id. The "sentiment_labels.txt" file consists of sentiment values for each phrase(each sentence consists of multiple phrases).

So if we want to get the sentiment values of a sentence we need to look into the phrase which is equal to the sentence. This functionality of getting a sentiment values and converting those values into 5 labels is shown below.

    # reading two csv files
    data_sentence = pd.read_csv('datasets/datasetSentences.txt',sep='\t')
    data_split = pd.read_csv('datasets/datasetSplit.txt')
    dictionary = pd.read_csv('datasets/dictionary.txt',sep='|')
    sentiment_labels = pd.read_csv('datasets/sentiment_labels.txt',sep='|')
  
    # Merging the "datasetSentences.txt" and "datasetSplit.txt" based on sentence_index
    data_sent_split = pd.merge(data_sentence,data_split,how='inner',on='sentence_index')
    
    #Merging data_sent_split with dictinary based on the phrase
    comp_data = data_sent_split.merge(dictionary, left_on='sentence', right_on='phrase')  
    
    # Merging comp_data and sentiment labels based on phrase_id
    sst_dataset = comp_data.merge(sentiment_labels,left_on='phrase_id', right_on='phrase ids')

    # Converting the floating point sentiment values to 5 Class labels
    sst_dataset['new_label'] = sst_dataset['sentiment values'].apply(convert_values)

    # Dropping the unwanted columns fron the dataframe
    sst_dataset.drop(['phrase ids','phrase','sentence_index','phrase_id','splitset_label','sentiment values'],axis=1,inplace=True)

#### Sample data
  ![cr1](https://user-images.githubusercontent.com/36162708/122920382-f97d2380-d37e-11eb-949c-23123ffceb9d.jpg)
    
    
 ### Training logs
 
     Epoch: 0 Train Loss: 1.614 | Train Acc: 16.54%  Val. Loss: 1.612 |  Val. Acc: 16.64% 
     Epoch: 1 Train Loss: 1.612 | Train Acc: 18.08%  Val. Loss: 1.611 |  Val. Acc: 17.67% 
     Epoch: 2 Train Loss: 1.609 | Train Acc: 20.51%  Val. Loss: 1.609 |  Val. Acc: 19.54%
     Epoch: 3 Train Loss: 1.607 | Train Acc: 22.99%  Val. Loss: 1.608 |  Val. Acc: 21.11%
     Epoch: 4 Train Loss: 1.604 | Train Acc: 25.13%  Val. Loss: 1.606 |  Val. Acc: 22.05%
     Epoch: 5 Train Loss: 1.601 | Train Acc: 26.62%  Val. Loss: 1.604 |  Val. Acc: 24.05% 
      . 
      . 
      . 
      . 
      . 
      Epoch: 45	Train Loss: 1.353 | Train Acc: 59.16%  Val. Loss: 1.536 |  Val. Acc: 34.18%
      Epoch: 46	Train Loss: 1.347 | Train Acc: 59.58%  Val. Loss: 1.536 |  Val. Acc: 34.21%
      Epoch: 47	Train Loss: 1.340 | Train Acc: 60.27%  Val. Loss: 1.535 |  Val. Acc: 34.30%
      Epoch: 48	Train Loss: 1.333 | Train Acc: 61.48%  Val. Loss: 1.534 |  Val. Acc: 33.93%
      Epoch: 49	Train Loss: 1.329 | Train Acc: 61.76%  Val. Loss: 1.535 |  Val. Acc: 34.10% 
      
### Results

      Minimum Train Loss: 1.329   
      Minimum Validation Loss: 1.534
   
   
      Maximum Train Accuracy: 61.76 %   
      Maximum Validation Accuracy: 34.76%  
    

![crc2](https://user-images.githubusercontent.com/36162708/122938405-5af9be00-d390-11eb-9804-c9141f48fb5a.jpg)

### Outcomes

    Sample Text: The invincible Werner Herzog is alive and well and living in LA
    Actual Value: Positive
    Predicted Value: Negative
    

    Sample Text: Just the labour involved in creating the layered richness of the imagery in this chiaroscuro of madness and light is astonishing
    Actual Value: Very Positive
    Predicted Value: Negative
    

    Sample Text: A truly moving experience and a perfect example of how art  when done right  can help heal clarify and comfort
    Actual Value: Very Positive
    Predicted Value: Negative
    

    Sample Text: Son of the Bride may be a good half-hour too long but comes replete with a flattering sense of mystery and quietness
    Actual Value: Very Negative
    Predicted Value: Very Negative
    

    Sample Text: A few artsy flourishes aside Narc is as gritty as a movie gets these days
    Actual Value: Neutral
    Predicted Value: Neutral
    

    Sample Text: 4 friends 2 couples 2000 miles and all the Pabst Blue Ribbon beer they can drink it s the ultimate redneck road-trip
    Actual Value: Positive
    Predicted Value: Negative


    Sample Text: Hip-hop has a history and it s a metaphor for this love story
    Actual Value: Neutral
    Predicted Value: Negative
    

    Sample Text: In Fessenden s horror trilogy this theme has proved important to him and is especially so in the finale
    Actual Value: Positive
    Predicted Value: Negative
    

    Sample Text: In between all the emotional seesawing it s hard to figure the depth of these two literary figures and even the times in which they lived
    Actual Value: Negative
    Predicted Value: Very Negative
    

    Sample Text: It s a great deal of sizzle and very little steak
    Actual Value: Very Negative
    Predicted Value: Very Negative
    

 #### Team Members
   1. Anish V
   2. Vimal Kaur
   3. Hari
   4. Nilanjana Dev Nath
