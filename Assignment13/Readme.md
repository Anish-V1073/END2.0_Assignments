# Assignment 13

## Objective
This [code](https://colab.research.google.com/github/bentrevett/pytorch-seq2seq/blob/master/6%20-%20Attention%20is%20All%20You%20Need.ipynb#scrollTo=FqXbPB80r8p4) is from the same repo that we were following.  

Your assignment is to remove all the legacy stuff from this and submit:

* Last 5 Training EPOCH logs
* Sample translation for 10 example
* Link to your repo

## Solution

### Training logs

    Epoch: 20 | Time: 0m 25s
      Train Loss: 2.015 | Train PPL:   7.502
       Val. Loss: 2.443 |  Val. PPL:  11.502
    Epoch: 21 | Time: 0m 25s
      Train Loss: 1.947 | Train PPL:   7.009
       Val. Loss: 2.413 |  Val. PPL:  11.171
    Epoch: 22 | Time: 0m 25s
      Train Loss: 1.885 | Train PPL:   6.584
       Val. Loss: 2.394 |  Val. PPL:  10.957
    Epoch: 23 | Time: 0m 25s
      Train Loss: 1.825 | Train PPL:   6.200
       Val. Loss: 2.368 |  Val. PPL:  10.676
    Epoch: 24 | Time: 0m 25s
      Train Loss: 1.768 | Train PPL:   5.858
       Val. Loss: 2.347 |  Val. PPL:  10.449
    Epoch: 25 | Time: 0m 25s
      Train Loss: 1.714 | Train PPL:   5.551
       Val. Loss: 2.337 |  Val. PPL:  10.347
       
 ### Sample outcomes
 
    src = ['Männer', ',', 'die', 'Volleyball', 'spielen', ',', 'wobei', 'ein', 'Mann', 'denn', 'Ball', 'nicht', 'trifft', 'während', 'seine', 'Hände', 'immer', 'noch', 'in', 'der', 'Luft', 'sind', '.', '\n']
    trg = ['Men', 'playing', 'volleyball', ',', 'with', 'one', 'player', 'missing', 'the', 'ball', 'but', 'hands', 'still', 'in', 'the', 'air', '.', '\n']
    predicted trg = ['Men', ',', 'one', 'man', 'playing', 'volleyball', ',', 'with', 'the', 'ball', ',', 'while', 'his', 'hands', 'in', 'the', 'air', ',', 'one', 'man', 'who', 'is', 'practicing', 'his', 'hands', ',', 'who', 'is', 'practicing', 'his', 'hands', ',', 'who', 'is', 'practicing', 'his', 'hands', ',', 'who', 'are', 'practicing', 'his', 'hands', ',', 'as', 'he', 'has', 'one', 'man', ',']


    src = ['Eine', 'Frau', ',', 'die', 'in', 'einer', 'Küche', 'eine', 'Schale', 'mit', 'Essen', 'hält', '.', '\n']
    trg = ['A', 'woman', 'holding', 'a', 'bowl', 'of', 'food', 'in', 'a', 'kitchen', '.', '\n']
    predicted trg = ['A', 'woman', 'holding', 'a', 'kitchen', 'with', 'a', 'microphone', 'holding', 'food', '.', '<eos>']


    src = ['Ein', 'sitzender', 'Mann', ',', 'der', 'an', 'einem', 'Tisch', 'in', 'seinem', 'Haus', 'mit', 'einem', 'Werkzeug', 'arbeitet', '.', '\n']
    trg = ['Man', 'sitting', 'using', 'tool', 'at', 'a', 'table', 'in', 'his', 'home', '.', '\n']
    predicted trg = ['A', 'man', 'sitting', 'at', 'a', 'table', 'with', 'his', 'metal', 'working', 'on', 'a', 'house', 'with', 'his', 'metal', 'work', '.', '<eos>']


    src = ['Drei', 'Leute', 'sitzen', 'in', 'einer', 'Höhle', '.', '\n']
    trg = ['Three', 'people', 'sit', 'in', 'a', 'cave', '.', '\n']
    predicted trg = ['Three', 'people', 'sitting', 'in', 'a', 'lake', '.', '<eos>']


    src = ['Ein', 'Mädchen', 'in', 'einem', 'Jeanskleid', 'läuft', 'über', 'einen', 'erhöhten', 'Schwebebalken', '.', '\n']
    trg = ['A', 'girl', 'in', 'a', 'jean', 'dress', 'is', 'walking', 'along', 'a', 'raised', 'balance', 'beam', '.', '\n']
    predicted trg = ['A', 'girl', 'in', 'a', 'dress', 'is', 'running', 'over', 'a', 'board', 'board', '.', '<eos>']


    src = ['Eine', 'Blondine', 'hält', 'mit', 'einem', 'Mann', 'im', 'Sand', 'Händchen', '.', '\n']
    trg = ['A', 'blond', 'holding', 'hands', 'with', 'a', 'guy', 'in', 'the', 'sand', '.', '\n']
    predicted trg = ['A', 'woman', 'holding', 'hands', 'in', 'the', 'sand', 'with', 'a', 'man', 'in', 'the', 'sand', '.', '<eos>']


    src = ['Eine', 'Frau', 'in', 'einem', 'Grauen', 'Pulli', 'und', 'mit', 'einer', 'schwarzen', 'Baseballmütze', 'steht', 'in', 'einem', 'Geschäft', 'in', 'der', 'Schlange', '.', '\n']
    trg = ['A', 'woman', 'in', 'a', 'gray', 'sweater', 'and', 'black', 'baseball', 'cap', 'is', 'standing', 'in', 'line', 'at', 'a', 'shop', '.', '\n']
    predicted trg = ['A', 'woman', 'in', 'a', 'yellow', 'sweater', 'and', 'black', 'cap', 'is', 'standing', 'in', 'a', 'store', 'with', 'a', 'black', 'cap', 'standing', 'in', 'a', 'store', '.', '<eos>']


    src = ['Die', 'Person', 'im', 'gestreiften', 'Hemd', 'klettert', 'auf', 'einen', 'Berg', '.', '\n']
    trg = ['The', 'person', 'in', 'the', 'striped', 'shirt', 'is', 'mountain', 'climbing', '.', '\n']
    predicted trg = ['The', 'person', 'in', 'the', 'striped', 'shirt', 'is', 'climbing', 'a', 'mountain', '.', '<eos>']


    src = ['Zwei', 'Männer', 'tun', 'so', 'als', 'seien', 'sie', 'Statuen', ',', 'während', 'Frauen', 'ihnen', 'zusehen', '.', '\n']
    trg = ['Two', 'men', 'pretend', 'to', 'be', 'statutes', 'while', 'women', 'look', 'on', '.', '\n']
    predicted trg = ['Two', 'men', 'watch', 'as', 'they', 'are', 'doing', 'their', 'instruments', 'while', 'them', 'watch', '.', '<eos>']


    src = ['Leute', ',', 'die', 'vor', 'einem', 'Gebäude', 'stehen', '.', '\n']
    trg = ['People', 'standing', 'outside', 'of', 'a', 'building', '.', '\n']
    predicted trg = ['People', 'standing', 'in', 'front', 'of', 'a', 'building', 'standing', 'in', 'front', 'of', 'a', 'yellow', 'building', '.', '<eos>']
