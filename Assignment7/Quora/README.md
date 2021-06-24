# Seq2Seq

## Objective
* Train model we wrote in the class on the following two datasets taken from this link (Links to an external site.): 
   * http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.)
   * https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (Links to an external site.)
* Once done, please upload the file to GitHub and proceed to answer these questions in the S7 - Assignment Solutions, where these questions are asked:
   * Share the link to your GitHub repo (100 pts for code quality/file structure/model accuracy) (100 pts)
   * Share the link to your readme file (100 points for proper readme file), this file can be the second part of your Part 1 Readme (basically you can have only 1 Readme, describing both assignments if you want) (100 pts)
   * Copy-paste the code related to your dataset preparation for both datasets.  (100 pts)
##  Quora Dataset (http://www.cs.cmu.edu/~ark/QA-data/)
Quora dataset consists of over 400,000 lines of potential question duplicate pairs. Each line contains IDs for each question in the pair, the full text for each question, and a binary value that indicates whether the line truly contains a duplicate pair.

After Removing a duplicate pairs from the original dataset, we have ~149000 questions pairs. Below is the code used to prepare the dataset

#### code
    # Reading the dataset 
    data_quora = pd.read_csv('datasets/quora_duplicate_questions.tsv',sep='\t')
  
    # Removing the duplicate entries from the dataframe
    data_quora.drop(data_quora[data_quora['is_duplicate'] == 0 ].index,inplace=True)
    
    # Dropping the unwanted columns from the dataframe
    data_quora.drop(['id','qid1','qid2','is_duplicate'],axis=1,inplace=True)

    data_quora.to_csv( "datasets/quora_dataset.csv", index=False, encoding='utf-8-sig')

#### Dataset sample
  ![cr4](https://user-images.githubusercontent.com/36162708/123104554-ad9cae00-d454-11eb-8432-28ebf18a544a.jpg)


### Training logs
 
     Epoch: 01 | Time: 5m 7s	Train Loss: 4.511 | Train PPL:  91.011	 Val. Loss: 4.313 |  Val. PPL:  74.644
     Epoch: 02 | Time: 5m 6s	Train Loss: 3.438 | Train PPL:  31.119	 Val. Loss: 3.870 |  Val. PPL:  47.937
     Epoch: 03 | Time: 5m 6s	Train Loss: 2.961 | Train PPL:  19.315	 Val. Loss: 3.728 |  Val. PPL:  41.610
     Epoch: 04 | Time: 5m 7s	Train Loss: 2.675 | Train PPL:  14.518	 Val. Loss: 3.612 |  Val. PPL:  37.057
     Epoch: 05 | Time: 5m 8s	Train Loss: 2.473 | Train PPL:  11.860	 Val. Loss: 3.531 |  Val. PPL:  34.144
     Epoch: 06 | Time: 5m 9s	Train Loss: 2.338 | Train PPL:  10.362	 Val. Loss: 3.524 |  Val. PPL:  33.928
     Epoch: 07 | Time: 5m 9s	Train Loss: 2.225 | Train PPL:   9.257	 Val. Loss: 3.558 |  Val. PPL:  35.079
     Epoch: 08 | Time: 5m 9s	Train Loss: 2.127 | Train PPL:   8.392	 Val. Loss: 3.564 |  Val. PPL:  35.306
     Epoch: 09 | Time: 5m 9s	Train Loss: 2.050 | Train PPL:   7.766	 Val. Loss: 3.551 |  Val. PPL:  34.834
     Epoch: 10 | Time: 5m 9s	Train Loss: 1.971 | Train PPL:   7.175	 Val. Loss: 3.567 |  Val. PPL:  35.393
     
### Results

      Minimum Train Loss: 1.971   
      Minimum Validation Loss: 3.524   
   
      Minimum Train PPL: 7.175   
      Minimum Valid PPL: 33.928 
    

