# Seq2Seq

## Objective
* Train model we wrote in the class on the following two datasets taken from this link (Links to an external site.): 
   * http://www.cs.cmu.edu/~ark/QA-data/ (Links to an external site.)
   * https://quoradata.quora.com/First-Quora-Dataset-Release-Question-Pairs (Links to an external site.)
* Once done, please upload the file to GitHub and proceed to answer these questions in the S7 - Assignment Solutions, where these questions are asked:
   * Share the link to your GitHub repo (100 pts for code quality/file structure/model accuracy) (100 pts)
   * Share the link to your readme file (100 points for proper readme file), this file can be the second part of your Part 1 Readme (basically you can have only 1 Readme, describing both assignments if you want) (100 pts)
   * Copy-paste the code related to your dataset preparation for both datasets.  (100 pts)
##  QA Dataset (http://www.cs.cmu.edu/~ark/QA-data/ 
There are three directories in the dataset, one for each year of students: S08, S09, and S10. The file "question_answer_pairs.txt" contains the questions and answers. The first line of the file contains column names for the tab-separated data fields in the file. This first line follows:

|ArticleTitle|Question|Answer|DifficultyFromQuestioner|DifficultyFromAnswerer|ArticleFile|
| :---: | :---: | :---: | :---: | :---: | :---: |

* ArticleTitle is the name of the Wikipedia article from which questions and answers initially came.
* Question is the question.<br>
* Answer is the answer.<br>
* DifficultyFromQuestioner is the prescribed difficulty rating for the question as given to the question-writer.<br> 
* DifficultyFromAnswerer is a difficulty rating assigned by the individual who evaluated and answered the question, 
which may differ from the difficulty in DifficultyFromQuestioner.<br>
* ArticleFile is the relative path to the prefix of the article files. html files (.htm) and cleaned 
text (.txt) files are provided.<br>

We combined S08,S09 and S10 question-answer pairs and removed unwanted columns(retaining only Question and Answer). In some of the questions-answer pairs the answer in Nan, we removed those entires and also the duplicate entries. Below is the code for dat preparation for QA datasets.

#### code
    # Reading S08,S09 and S10 question-answer pairs
    data_s08 = pd.read_csv('datasets/question_answer_pairs_S08.txt',sep='\t')
    data_s09 = pd.read_csv('datasets/question_answer_pairs_S09.txt',sep='\t')
    data_s10 = pd.read_csv('datasets/question_answer_pairs_S10.txt',sep='\t')

    # Combining S08 and S09 question-answer pairs
    df = data_s08.append(data_s09)
    
    # Combining S09 and S10 question-answer pairs
    combined = df.append(data_s10)

    # Removing all other columns except Question and Answer
    combined.drop(['DifficultyFromQuestioner','DifficultyFromAnswerer','ArticleFile','ArticleTitle'],axis=1,inplace=True)

    combined['Question'] = combined['Question'].str.replace('[.\']','')
    combined['Answer'] = combined['Answer'].str.replace('[.\']','')

    combined['Answer'] = combined['Answer'].str.lower()
    combined['Question'] = combined['Question'].str.lower()

    # Removing the duplicate entries from Question and Answer pairs
    combined.drop_duplicates(keep=False,inplace=True)
    combined['Answer'].replace('', np.nan, inplace=True)
    combined.dropna(subset=['Answer'], inplace=True)
    combined.to_csv( "datasets/qa_dataset.csv", index=False, encoding='utf-8-sig')
#### Dataset sample
  ![cr3](https://user-images.githubusercontent.com/36162708/123035297-bc5d7380-d408-11eb-8ee3-15af024afaf1.jpg)

### Training logs
 
     Epoch: 01 | Time: 0m 2s	Train Loss: 5.228 | Train PPL: 186.413	 Val. Loss: 3.943 |  Val. PPL:  51.578
     Epoch: 02 | Time: 0m 2s	Train Loss: 4.600 | Train PPL:  99.453	 Val. Loss: 3.888 |  Val. PPL:  48.799
     Epoch: 03 | Time: 0m 1s	Train Loss: 4.512 | Train PPL:  91.065	 Val. Loss: 3.839 |  Val. PPL:  46.485
     Epoch: 04 | Time: 0m 2s	Train Loss: 4.464 | Train PPL:  86.804	 Val. Loss: 3.874 |  Val. PPL:  48.144
     Epoch: 05 | Time: 0m 2s	Train Loss: 4.417 | Train PPL:  82.869	 Val. Loss: 3.841 |  Val. PPL:  46.586
      . 
      . 
      . 
      . 
      . 
      Epoch: 22 | Time: 0m 2s	Train Loss: 3.790 | Train PPL:  44.278	 Val. Loss: 3.886 |  Val. PPL:  48.726
      Epoch: 23 | Time: 0m 2s	Train Loss: 3.773 | Train PPL:  43.529	 Val. Loss: 3.896 |  Val. PPL:  49.206
      Epoch: 24 | Time: 0m 2s	Train Loss: 3.744 | Train PPL:  42.259	 Val. Loss: 3.936 |  Val. PPL:  51.196
      Epoch: 25 | Time: 0m 2s	Train Loss: 3.678 | Train PPL:  39.552	 Val. Loss: 3.947 |  Val. PPL:  51.800
### Results

      Minimum Train Loss: 3.628   
      Minimum Validation Loss: 3.832   
   
      Minimum Train PPL: 39.252   
      Minimum Valid PPL: 46.267 
    

 #### Team Members
   1. Anish V
   2. Vimal Kaur
   3. Hari
   4. Nilanjana Dev Nath
