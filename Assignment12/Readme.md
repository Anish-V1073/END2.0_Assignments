# Assignment 11

## Objective
Take the code from [Transformers from scratch](https://github.com/aladdinpersson/Machine-Learning-Collection/blob/a2ee9271b5280be6994660c7982d0f44c67c3b63/ML/Pytorch/more_advanced/transformer_from_scratch/transformer_from_scratch.py) , and make it work with any dataset. Submit the GitHub repo's ReadMe file, where I can see answers to these questions:

* what dataset you have used
* what problem have you solved (fill in the blank, translation, text generation, etc)
* the output of your training for 10 epochs
* you can work with your existing team, or the new team you are going to make below!

## Solution

**Dataset:** We are using the Multi30K dataset from **torchtext.datasets** 

**Problem:** Translation(i.e German to English Translation)

### Training Logs

    Epoch: 01 | Time: 1m 13s
      Train Loss: 6.190 | Train PPL: 487.840
      Val. Loss: 5.674 |  Val. PPL: 291.059
    Epoch: 02 | Time: 1m 15s
      Train Loss: 5.637 | Train PPL: 280.723
      Val. Loss: 5.679 |  Val. PPL: 292.792
    Epoch: 03 | Time: 1m 17s
      Train Loss: 5.634 | Train PPL: 279.667
      Val. Loss: 5.680 |  Val. PPL: 292.933
    Epoch: 04 | Time: 1m 18s
      Train Loss: 5.626 | Train PPL: 277.665
      Val. Loss: 5.700 |  Val. PPL: 298.920
    Epoch: 05 | Time: 1m 19s
      Train Loss: 5.629 | Train PPL: 278.355
      Val. Loss: 5.712 |  Val. PPL: 302.330
    Epoch: 06 | Time: 1m 19s
      Train Loss: 5.623 | Train PPL: 276.590
      Val. Loss: 5.883 |  Val. PPL: 359.062
    Epoch: 07 | Time: 1m 19s
      Train Loss: 5.645 | Train PPL: 282.796
      Val. Loss: 5.701 |  Val. PPL: 299.234
    Epoch: 08 | Time: 1m 19s
      Train Loss: 5.618 | Train PPL: 275.226
      Val. Loss: 5.698 |  Val. PPL: 298.165
    Epoch: 09 | Time: 1m 19s
	    Train Loss: 5.611 | Train PPL: 273.407
	    Val. Loss: 5.780 |  Val. PPL: 323.606
    Epoch: 10 | Time: 1m 19s
	    Train Loss: 5.609 | Train PPL: 273.007
	    Val. Loss: 5.782 |  Val. PPL: 324.261
