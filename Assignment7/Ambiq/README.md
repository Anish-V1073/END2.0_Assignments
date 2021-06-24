# Seq2Seq

##  AmbigQA Dataset (https://nlp.cs.washington.edu/ambigqa/) 
AmbigQA, a new open-domain question answering task that consists of predicting a set of question and answer pairs, where each plausible answer is associated with a disambiguated rewriting of the original question. A data set covering 14,042 open-ended QI-open questions.

We have choosen the AmbigQA light version dataset. The train and test dataset in the AmbigQA is in the JSON format. Below are the some of the data samples in JSON format.

    {
        "annotations": [            
            {
                "type": "singleAnswer",
                "answer": [
                    "October of that year",
                    "October 1919"
                ]
            }
        ],
        "id": "-6975273415196871312",
        "question": "When was the first airline meal served during a flight?"
    },
    
    {
        "annotations": [
            {
                "type": "multipleQAs",
                "qaPairs": [
                    {
                        "question": "Who is the 4th chairman of african union commission?",
                        "answer": [
                            "Moussa Faki",
                            "Moussa Faki Mahamat"
                        ]
                    },
                    {
                        "question": "Who is the 3rd chairman of african union commission?",
                        "answer": [
                            "Nkosazana Clarice Dlamini-Zuma",
                            "Nkosazana Dlamini-Zuma"
                        ]
                    },
                    {
                        "question": "Who is the 2nd chairman of african union commission?",
                        "answer": [
                            "Jean Ping"
                        ]
                    }
                ]
     }
     
We parsed the the above JSON data and obtained a question-answer pair.

#### code
    def prepare(data,name):
      que_ans_list = list()
      filename = "%s_data.csv" % name
      for i in range(len(data)):
        # Selecting the type 'multipleQAs'
        if data[i]['annotations'][0]['type'] == 'multipleQAs':
          temp = data[i]['annotations'][0]['qaPairs']
          for j in range(len(temp)):
            que_ans_list.append([temp[j]['question'],temp[j]['answer'][0]])
        # Selecting the type 'singleAnswer'
        elif data[i]['annotations'][0]['type'] == 'singleAnswer':
          que_ans_list.append([data[i]['question'],data[i]['annotations'][0]['answer'][0]])
      df = pd.DataFrame(que_ans_list, columns =['Questions', 'Answers'])
      df.to_csv(filename,sep='\t',index=False)

#### Dataset sample
  ![cr5](https://user-images.githubusercontent.com/36162708/123113890-8649df00-d45c-11eb-95e6-2965c4036282.jpg)

### Training logs
 
     Epoch: 01 | Time: 1m 30s	Train Loss: 4.997 | Train PPL: 147.981	 Val. Loss: 4.126 |  Val. PPL:  61.946
     Epoch: 02 | Time: 1m 29s	Train Loss: 4.424 | Train PPL:  83.420	 Val. Loss: 4.010 |  Val. PPL:  55.158
     Epoch: 03 | Time: 1m 30s	Train Loss: 4.231 | Train PPL:  68.767	 Val. Loss: 3.960 |  Val. PPL:  52.435
     Epoch: 04 | Time: 1m 30s	Train Loss: 4.102 | Train PPL:  60.478	 Val. Loss: 3.947 |  Val. PPL:  51.786
     Epoch: 05 | Time: 1m 30s	Train Loss: 3.977 | Train PPL:  53.357	 Val. Loss: 3.906 |  Val. PPL:  49.715
     Epoch: 06 | Time: 1m 30s	Train Loss: 3.834 | Train PPL:  46.240	 Val. Loss: 3.899 |  Val. PPL:  49.340
     Epoch: 07 | Time: 1m 30s	Train Loss: 3.693 | Train PPL:  40.151	 Val. Loss: 3.927 |  Val. PPL:  50.751
     Epoch: 08 | Time: 1m 30s	Train Loss: 3.569 | Train PPL:  40.487	 Val. Loss: 3.951 |  Val. PPL:  50.012
     Epoch: 09 | Time: 1m 30s	Train Loss: 3.645 | Train PPL:  40.121	 Val. Loss: 3.634 |  Val. PPL:  51.751
     Epoch: 10 | Time: 1m 30s	Train Loss: 3.423 | Train PPL:  41.234	 Val. Loss: 3.756 |  Val. PPL:  52.012
### Results

      Minimum Train Loss: 3.423   
      Minimum Validation Loss: 3.634   
   
      Minimum Train PPL: 40.121   
      Maximum Valid PPL: 49.715 
    


 #### Team Members
   1. Anish V
   2. Vimal Kaur
   3. Hari
   4. Nilanjana Dev Nath
