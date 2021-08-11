# Assignment 14

## Objective
* TASK 1: Train BERT using the code mentioned [here](https://drive.google.com/file/d/1Zp2_Uka8oGDYsSe5ELk-xz6wIX8OIkB7/view?usp=sharing) on the Squad Dataset for 20% overall samples (1/5 Epochs). Show results on 5 samples. 
* TASK 2: Reproductive [these](https://mccormickml.com/2019/07/22/BERT-fine-tuning/)  results, and show output on 5 samples.
* TASK 3: Reproduce the training explained in this [blog](https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c). You can decide to pick fewer datasets. 
* Proceed to Session 14 - Assignment Solutions page and:
    * Submit README link for Task 1 (training log snippets and 5 sample results along with BERT description must be available) - 750
    * Submit README link for Task 2 (training log snippets and 5 sample results) - 250
    * Submit README link for Task 3 (training log snippets and 5 sample results along with BART description must be available) - 1000

## Solutions

   * [Task 1](#task1)
   * [Task 2](#task2)
   * [Task 3](#task3)

<a id="task1"></a>
# Task 1

### BERT
BERT stands for Bidirectional Encoder Representations from Transformers. It is designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context. As a result, the pre-trained BERT model can be fine-tuned with just one additional output layer to create state-of-the-art models for a wide range of NLP tasks.

BERT makes use of Transformer architecture (attention mechanism) that learns contextual relations between words in a text. In its vanilla form, Transformer includes two separate mechanisms - an encoder that reads the text input and a decoder that produces a prediction for the task.

BERT is released in two sizes BERTBASE and BERTLARGE. The BASE model is used to measure the performance of the architecture comparable to another architecture and the LARGE model produces state-of-the-art results that were reported in the research paper.

BERT is basically an Encoder stack of transformer architecture. A transformer architecture is an encoder-decoder network that uses self-attention on the encoder side and attention on the decoder side. BERTBASE has 12 layers in the Encoder stack while BERTLARGE has 24 layers in the Encoder stack. These are more than the Transformer architecture described in the original paper (6 encoder layers). BERT architectures (BASE and LARGE) also have larger feedforward-networks (768 and 1024 hidden units respectively), and more attention heads (12 and 16 respectively) than the Transformer architecture suggested in the original paper. It contains 512 hidden units and 8 attention heads. BERTBASE contains 110M parameters while BERTLARGE has 340M parameters.

We have used the Squad Dataset to try a training of Bert model in this task and we successfuly trained the model and got the following results.


### Training logs
      Epoch:   0%|          | 0/1 [00:00<?, ?it/s]

      ***** Running training *****
        Num examples = 28013
        Num Epochs = 1
        Batch size = 16
        Total optimization steps = 1750
      /usr/local/lib/python3.7/dist-packages/pytorch_transformers/optimization.py:166: UserWarning: This overload of add_ is deprecated:
         add_(Number alpha, Tensor other)
      Consider using one of the following signatures instead:
         add_(Tensor other, *, Number alpha) (Triggered internally at  /pytorch/torch/csrc/utils/python_arg_parser.cpp:1025.)
        exp_avg.mul_(beta1).add_(1.0 - beta1, grad)
      Iteration:   0%|          | 0/1750 [00:00<?, ?it/s]
      Iteration:   0%|          | 2/1750 [00:01<22:48,  1.28it/s]
      Iteration:   0%|          | 3/1750 [00:03<29:15,  1.00s/it]
      Iteration:   0%|          | 4/1750 [00:04<33:46,  1.16s/it]
      Iteration:   0%|          | 5/1750 [00:06<36:55,  1.27s/it]
      Iteration:   0%|          | 6/1750 [00:07<39:05,  1.34s/it]
      Iteration:   0%|          | 7/1750 [00:09<40:36,  1.40s/it]
      Iteration:   0%|          | 8/1750 [00:10<41:38,  1.43s/it]
      Iteration:   1%|          | 9/1750 [00:12<42:21,  1.46s/it]
      Iteration:   1%|          | 10/1750 [00:13<42:48,  1.48s/it]
      Iteration:   1%|          | 11/1750 [00:15<43:05,  1.49s/it]
      Iteration:   1%|          | 12/1750 [00:16<43:23,  1.50s/it]
      Iteration:   1%|          | 13/1750 [00:18<43:34,  1.51s/it]
      Iteration:   1%|          | 14/1750 [00:19<43:43,  1.51s/it]
      Iteration:   1%|          | 15/1750 [00:21<43:52,  1.52s/it]
      Iteration:   1%|          | 16/1750 [00:22<43:50,  1.52s/it]
      Iteration:   1%|          | 17/1750 [00:24<43:56,  1.52s/it]
      Iteration:   1%|          | 18/1750 [00:25<43:56,  1.52s/it]
      Iteration:   1%|          | 19/1750 [00:27<43:50,  1.52s/it]
      Iteration:   1%|          | 20/1750 [00:28<43:49,  1.52s/it]
      Iteration:   1%|          | 21/1750 [00:30<43:48,  1.52s/it]
      Iteration:   1%|▏         | 22/1750 [00:31<43:48,  1.52s/it]
      Iteration:   1%|▏         | 23/1750 [00:33<43:48,  1.52s/it]
      Iteration:   1%|▏         | 24/1750 [00:35<43:48,  1.52s/it]
      Iteration:   1%|▏         | 25/1750 [00:36<43:45,  1.52s/it]
      Iteration:   1%|▏         | 26/1750 [00:38<43:42,  1.52s/it]
      Iteration:   2%|▏         | 27/1750 [00:39<43:35,  1.52s/it]
      Iteration:   2%|▏         | 28/1750 [00:41<43:30,  1.52s/it]
      Iteration:   2%|▏         | 29/1750 [00:42<43:29,  1.52s/it]
      Iteration:   2%|▏         | 30/1750 [00:44<43:28,  1.52s/it]
      Iteration:   2%|▏         | 31/1750 [00:45<43:24,  1.51s/it]
      Iteration:   2%|▏         | 32/1750 [00:47<43:16,  1.51s/it]
      Iteration:   2%|▏         | 33/1750 [00:48<43:12,  1.51s/it]
      Iteration:   2%|▏         | 34/1750 [00:50<43:13,  1.51s/it]
      Iteration:   2%|▏         | 35/1750 [00:51<43:12,  1.51s/it]
      Iteration:   2%|▏         | 36/1750 [00:53<43:08,  1.51s/it]
      Iteration:   2%|▏         | 37/1750 [00:54<43:07,  1.51s/it]
      Iteration:   2%|▏         | 38/1750 [00:56<43:05,  1.51s/it]
      Iteration:   2%|▏         | 39/1750 [00:57<43:02,  1.51s/it]
      Iteration:   2%|▏         | 40/1750 [00:59<43:00,  1.51s/it]
      Iteration:   2%|▏         | 41/1750 [01:00<43:04,  1.51s/it]
      Iteration:   2%|▏         | 42/1750 [01:02<42:59,  1.51s/it]
      Iteration:   2%|▏         | 43/1750 [01:03<42:59,  1.51s/it]
      Iteration:   3%|▎         | 44/1750 [01:05<42:56,  1.51s/it]
      Iteration:   3%|▎         | 45/1750 [01:06<42:52,  1.51s/it]
      Iteration:   3%|▎         | 46/1750 [01:08<42:49,  1.51s/it]
      Iteration:   3%|▎         | 47/1750 [01:09<42:45,  1.51s/it]
      Iteration:   3%|▎         | 48/1750 [01:11<42:54,  1.51s/it]
      Iteration:   3%|▎         | 49/1750 [01:12<42:52,  1.51s/it]
      Iteration:   3%|▎         | 50/1750 [01:14<42:48,  1.51s/it]
      Iteration:   3%|▎         | 51/1750 [01:15<42:45,  1.51s/it]
      Iteration:   3%|▎         | 52/1750 [01:17<42:40,  1.51s/it]
      Iteration:   3%|▎         | 53/1750 [01:18<42:38,  1.51s/it]
      Iteration:   3%|▎         | 54/1750 [01:20<42:38,  1.51s/it]
      Iteration:   3%|▎         | 55/1750 [01:21<42:35,  1.51s/it]
      Iteration:   3%|▎         | 56/1750 [01:23<42:32,  1.51s/it]
      Iteration:   3%|▎         | 57/1750 [01:24<42:34,  1.51s/it]
      Iteration:   3%|▎         | 58/1750 [01:26<42:35,  1.51s/it]
      Iteration:   3%|▎         | 59/1750 [01:27<42:33,  1.51s/it]
      Iteration:   3%|▎         | 60/1750 [01:29<42:28,  1.51s/it]
      Iteration:   3%|▎         | 61/1750 [01:30<42:27,  1.51s/it]
      Iteration:   4%|▎         | 62/1750 [01:32<42:22,  1.51s/it]
      Iteration:   4%|▎         | 63/1750 [01:33<42:22,  1.51s/it]
      Iteration:   4%|▎         | 64/1750 [01:35<42:20,  1.51s/it]
      Iteration:   4%|▎         | 65/1750 [01:36<42:23,  1.51s/it]
      Iteration:   4%|▍         | 66/1750 [01:38<42:19,  1.51s/it]
      .
      .
      .
      .
      .
      .
      .
      Iteration:  57%|█████▋    | 994/1750 [24:53<18:56,  1.50s/it]
      Iteration:  57%|█████▋    | 995/1750 [24:54<18:55,  1.50s/it]
      Iteration:  57%|█████▋    | 996/1750 [24:56<18:53,  1.50s/it]
      Iteration:  57%|█████▋    | 997/1750 [24:57<18:50,  1.50s/it]
      Iteration:  57%|█████▋    | 998/1750 [24:59<18:50,  1.50s/it]
      Iteration:  57%|█████▋    | 999/1750 [25:00<18:48,  1.50s/it]
      Iteration:  57%|█████▋    | 1000/1750 [25:02<18:46,  1.50s/it]

      Train loss: 1.7036315193772316
      Saving model checkpoint to /content/drive/MyDrive/SQuAD/checkpoint-1000

      Iteration:  57%|█████▋    | 1001/1750 [25:05<26:32,  2.13s/it]
      Iteration:  57%|█████▋    | 1002/1750 [25:07<24:09,  1.94s/it]
      Iteration:  57%|█████▋    | 1003/1750 [25:08<22:36,  1.82s/it]
      Iteration:  57%|█████▋    | 1004/1750 [25:10<21:22,  1.72s/it]
      Iteration:  57%|█████▋    | 1005/1750 [25:11<20:32,  1.65s/it]
      Iteration:  57%|█████▋    | 1006/1750 [25:13<19:57,  1.61s/it]
      Iteration:  58%|█████▊    | 1007/1750 [25:14<19:32,  1.58s/it]
      Iteration:  58%|█████▊    | 1008/1750 [25:16<19:13,  1.56s/it]
      Iteration:  58%|█████▊    | 1009/1750 [25:17<19:00,  1.54s/it]
      Iteration:  58%|█████▊    | 1010/1750 [25:19<18:52,  1.53s/it]
      Iteration:  58%|█████▊    | 1011/1750 [25:20<18:48,  1.53s/it]
      Iteration:  58%|█████▊    | 1012/1750 [25:22<18:42,  1.52s/it]
      Iteration:  58%|█████▊    | 1013/1750 [25:23<18:35,  1.51s/it]
      Iteration:  58%|█████▊    | 1014/1750 [25:25<18:31,  1.51s/it]
      Iteration:  58%|█████▊    | 1015/1750 [25:26<18:28,  1.51s/it]
      Iteration:  58%|█████▊    | 1016/1750 [25:28<18:25,  1.51s/it]
      Iteration:  58%|█████▊    | 1017/1750 [25:29<18:23,  1.51s/it]
      Iteration:  58%|█████▊    | 1018/1750 [25:31<18:20,  1.50s/it]
      Iteration:  58%|█████▊    | 1019/1750 [25:32<18:17,  1.50s/it]
      Iteration:  58%|█████▊    | 1020/1750 [25:34<18:16,  1.50s/it]
      Iteration:  58%|█████▊    | 1021/1750 [25:35<18:14,  1.50s/it]
      Iteration:  58%|█████▊    | 1022/1750 [25:37<18:11,  1.50s/it]
      Iteration:  58%|█████▊    | 1023/1750 [25:38<18:12,  1.50s/it]
      Iteration:  59%|█████▊    | 1024/1750 [25:40<18:10,  1.50s/it]
      Iteration:  59%|█████▊    | 1025/1750 [25:41<18:09,  1.50s/it]
      Iteration:  59%|█████▊    | 1026/1750 [25:43<18:08,  1.50s/it]
      Iteration:  59%|█████▊    | 1027/1750 [25:44<18:05,  1.50s/it]
      Iteration:  59%|█████▊    | 1028/1750 [25:46<18:05,  1.50s/it]
      Iteration:  59%|█████▉    | 1029/1750 [25:47<18:04,  1.50s/it]
      Iteration:  59%|█████▉    | 1030/1750 [25:49<18:03,  1.51s/it]
      Iteration:  59%|█████▉    | 1031/1750 [25:50<18:01,  1.50s/it]
      Iteration:  59%|█████▉    | 1032/1750 [25:52<18:00,  1.51s/it]
      .
      .
      .
      .
      .
      .
      .
      .
      Iteration:  93%|█████████▎| 1625/1750 [40:43<03:07,  1.50s/it]
      Iteration:  93%|█████████▎| 1626/1750 [40:45<03:06,  1.50s/it]
      Iteration:  93%|█████████▎| 1627/1750 [40:46<03:04,  1.50s/it]
      Iteration:  93%|█████████▎| 1628/1750 [40:48<03:03,  1.50s/it]
      Iteration:  93%|█████████▎| 1629/1750 [40:49<03:01,  1.50s/it]
      Iteration:  93%|█████████▎| 1630/1750 [40:51<03:00,  1.50s/it]
      Iteration:  93%|█████████▎| 1631/1750 [40:52<02:58,  1.50s/it]
      Iteration:  93%|█████████▎| 1632/1750 [40:54<02:57,  1.51s/it]
      Iteration:  93%|█████████▎| 1633/1750 [40:55<02:56,  1.51s/it]
      Iteration:  93%|█████████▎| 1634/1750 [40:57<02:54,  1.51s/it]
      Iteration:  93%|█████████▎| 1635/1750 [40:58<02:53,  1.51s/it]
      Iteration:  93%|█████████▎| 1636/1750 [41:00<02:51,  1.51s/it]
      Iteration:  94%|█████████▎| 1637/1750 [41:01<02:50,  1.51s/it]
      Iteration:  94%|█████████▎| 1638/1750 [41:03<02:48,  1.50s/it]
      Iteration:  94%|█████████▎| 1639/1750 [41:04<02:46,  1.50s/it]
      Iteration:  94%|█████████▎| 1640/1750 [41:06<02:45,  1.50s/it]
      Iteration:  94%|█████████▍| 1641/1750 [41:07<02:43,  1.50s/it]
      Iteration:  94%|█████████▍| 1642/1750 [41:09<02:42,  1.50s/it]
      Iteration:  94%|█████████▍| 1643/1750 [41:10<02:40,  1.50s/it]
      Iteration:  94%|█████████▍| 1644/1750 [41:12<02:39,  1.50s/it]
      Iteration:  94%|█████████▍| 1645/1750 [41:13<02:37,  1.50s/it]
      Iteration:  94%|█████████▍| 1646/1750 [41:15<02:35,  1.50s/it]
      Iteration:  94%|█████████▍| 1647/1750 [41:16<02:34,  1.50s/it]
      Iteration:  94%|█████████▍| 1648/1750 [41:18<02:33,  1.50s/it]
      Iteration:  94%|█████████▍| 1649/1750 [41:19<02:31,  1.50s/it]
      Iteration:  94%|█████████▍| 1650/1750 [41:21<02:30,  1.50s/it]
      Iteration:  94%|█████████▍| 1651/1750 [41:22<02:28,  1.50s/it]
      Iteration:  94%|█████████▍| 1652/1750 [41:24<02:26,  1.50s/it]
      Iteration:  94%|█████████▍| 1653/1750 [41:25<02:25,  1.50s/it]
      Iteration:  95%|█████████▍| 1654/1750 [41:27<02:23,  1.49s/it]
      Iteration:  95%|█████████▍| 1655/1750 [41:28<02:22,  1.50s/it]
      Iteration:  95%|█████████▍| 1656/1750 [41:30<02:20,  1.49s/it]
      Iteration:  95%|█████████▍| 1657/1750 [41:31<02:18,  1.49s/it]
      Iteration:  95%|█████████▍| 1658/1750 [41:33<02:17,  1.49s/it]
      Iteration:  95%|█████████▍| 1659/1750 [41:34<02:15,  1.49s/it]
      Iteration:  95%|█████████▍| 1660/1750 [41:36<02:14,  1.49s/it]
      Iteration:  95%|█████████▍| 1661/1750 [41:37<02:12,  1.49s/it]
      Iteration:  95%|█████████▍| 1662/1750 [41:39<02:11,  1.49s/it]
      Iteration:  95%|█████████▌| 1663/1750 [41:40<02:10,  1.50s/it]
      Iteration:  95%|█████████▌| 1664/1750 [41:42<02:08,  1.50s/it]
      Iteration:  95%|█████████▌| 1665/1750 [41:43<02:07,  1.50s/it]
      Iteration:  95%|█████████▌| 1666/1750 [41:45<02:06,  1.50s/it]
      Iteration:  95%|█████████▌| 1667/1750 [41:46<02:04,  1.50s/it]
      Iteration:  95%|█████████▌| 1668/1750 [41:48<02:03,  1.50s/it]
      Iteration:  95%|█████████▌| 1669/1750 [41:49<02:01,  1.50s/it]
      Iteration:  95%|█████████▌| 1670/1750 [41:51<02:00,  1.50s/it]
      Iteration:  95%|█████████▌| 1671/1750 [41:52<01:58,  1.50s/it]
      Iteration:  96%|█████████▌| 1672/1750 [41:54<01:57,  1.50s/it]
      Iteration:  96%|█████████▌| 1673/1750 [41:55<01:55,  1.50s/it]
      Iteration:  96%|█████████▌| 1674/1750 [41:57<01:53,  1.50s/it]
      Iteration:  96%|█████████▌| 1675/1750 [41:58<01:52,  1.50s/it]
      Iteration:  96%|█████████▌| 1676/1750 [42:00<01:50,  1.50s/it]
      Iteration:  96%|█████████▌| 1677/1750 [42:01<01:49,  1.50s/it]
      Iteration:  96%|█████████▌| 1678/1750 [42:03<01:47,  1.50s/it]
      Iteration:  96%|█████████▌| 1679/1750 [42:04<01:46,  1.50s/it]
      Iteration:  96%|█████████▌| 1680/1750 [42:06<01:45,  1.50s/it]
      Iteration:  96%|█████████▌| 1681/1750 [42:07<01:43,  1.50s/it]
      Iteration:  96%|█████████▌| 1682/1750 [42:09<01:42,  1.50s/it]
      Iteration:  96%|█████████▌| 1683/1750 [42:10<01:40,  1.50s/it]
      Iteration:  96%|█████████▌| 1684/1750 [42:12<01:39,  1.50s/it]
      Iteration:  96%|█████████▋| 1685/1750 [42:13<01:37,  1.50s/it]
      Iteration:  96%|█████████▋| 1686/1750 [42:15<01:36,  1.50s/it]
      Iteration:  96%|█████████▋| 1687/1750 [42:16<01:34,  1.50s/it]
      Iteration:  96%|█████████▋| 1688/1750 [42:18<01:33,  1.50s/it]
      Iteration:  97%|█████████▋| 1689/1750 [42:19<01:31,  1.50s/it]
      Iteration:  97%|█████████▋| 1690/1750 [42:21<01:30,  1.50s/it]
      Iteration:  97%|█████████▋| 1691/1750 [42:22<01:28,  1.50s/it]
      Iteration:  97%|█████████▋| 1692/1750 [42:24<01:27,  1.50s/it]
      Iteration:  97%|█████████▋| 1693/1750 [42:25<01:25,  1.50s/it]
      Iteration:  97%|█████████▋| 1694/1750 [42:27<01:23,  1.50s/it]
      Iteration:  97%|█████████▋| 1695/1750 [42:28<01:22,  1.50s/it]
      Iteration:  97%|█████████▋| 1696/1750 [42:30<01:20,  1.50s/it]
      Iteration:  97%|█████████▋| 1697/1750 [42:31<01:19,  1.50s/it]
      Iteration:  97%|█████████▋| 1698/1750 [42:33<01:18,  1.50s/it]
      Iteration:  97%|█████████▋| 1699/1750 [42:34<01:16,  1.50s/it]
      Iteration:  97%|█████████▋| 1700/1750 [42:36<01:15,  1.50s/it]
      Iteration:  97%|█████████▋| 1701/1750 [42:37<01:13,  1.51s/it]
      Iteration:  97%|█████████▋| 1702/1750 [42:39<01:12,  1.51s/it]
      Iteration:  97%|█████████▋| 1703/1750 [42:40<01:10,  1.50s/it]
      Iteration:  97%|█████████▋| 1704/1750 [42:42<01:09,  1.51s/it]
      Iteration:  97%|█████████▋| 1705/1750 [42:43<01:07,  1.50s/it]
      Iteration:  97%|█████████▋| 1706/1750 [42:45<01:06,  1.51s/it]
      Iteration:  98%|█████████▊| 1707/1750 [42:46<01:04,  1.50s/it]
      Iteration:  98%|█████████▊| 1708/1750 [42:48<01:03,  1.50s/it]
      Iteration:  98%|█████████▊| 1709/1750 [42:49<01:01,  1.51s/it]
      Iteration:  98%|█████████▊| 1710/1750 [42:51<01:00,  1.50s/it]
      Iteration:  98%|█████████▊| 1711/1750 [42:52<00:58,  1.50s/it]
      Iteration:  98%|█████████▊| 1712/1750 [42:54<00:57,  1.51s/it]
      Iteration:  98%|█████████▊| 1713/1750 [42:55<00:55,  1.51s/it]
      Iteration:  98%|█████████▊| 1714/1750 [42:57<00:54,  1.51s/it]
      Iteration:  98%|█████████▊| 1715/1750 [42:58<00:52,  1.51s/it]
      Iteration:  98%|█████████▊| 1716/1750 [43:00<00:51,  1.51s/it]
      Iteration:  98%|█████████▊| 1717/1750 [43:01<00:49,  1.51s/it]
      Iteration:  98%|█████████▊| 1718/1750 [43:03<00:48,  1.50s/it]
      Iteration:  98%|█████████▊| 1719/1750 [43:04<00:46,  1.51s/it]
      Iteration:  98%|█████████▊| 1720/1750 [43:06<00:45,  1.51s/it]
      Iteration:  98%|█████████▊| 1721/1750 [43:07<00:43,  1.51s/it]
      Iteration:  98%|█████████▊| 1722/1750 [43:09<00:42,  1.51s/it]
      Iteration:  98%|█████████▊| 1723/1750 [43:10<00:40,  1.51s/it]
      Iteration:  99%|█████████▊| 1724/1750 [43:12<00:39,  1.51s/it]
      Iteration:  99%|█████████▊| 1725/1750 [43:13<00:37,  1.51s/it]
      Iteration:  99%|█████████▊| 1726/1750 [43:15<00:36,  1.51s/it]
      Iteration:  99%|█████████▊| 1727/1750 [43:16<00:34,  1.51s/it]
      Iteration:  99%|█████████▊| 1728/1750 [43:18<00:33,  1.51s/it]
      Iteration:  99%|█████████▉| 1729/1750 [43:19<00:31,  1.51s/it]
      Iteration:  99%|█████████▉| 1730/1750 [43:21<00:30,  1.51s/it]
      Iteration:  99%|█████████▉| 1731/1750 [43:22<00:28,  1.51s/it]
      Iteration:  99%|█████████▉| 1732/1750 [43:24<00:27,  1.51s/it]
      Iteration:  99%|█████████▉| 1733/1750 [43:25<00:25,  1.51s/it]
      Iteration:  99%|█████████▉| 1734/1750 [43:27<00:24,  1.50s/it]
      Iteration:  99%|█████████▉| 1735/1750 [43:28<00:22,  1.51s/it]
      Iteration:  99%|█████████▉| 1736/1750 [43:30<00:21,  1.51s/it]
      Iteration:  99%|█████████▉| 1737/1750 [43:31<00:19,  1.51s/it]
      Iteration:  99%|█████████▉| 1738/1750 [43:33<00:18,  1.51s/it]
      Iteration:  99%|█████████▉| 1739/1750 [43:34<00:16,  1.51s/it]
      Iteration:  99%|█████████▉| 1740/1750 [43:36<00:15,  1.51s/it]
      Iteration:  99%|█████████▉| 1741/1750 [43:37<00:13,  1.50s/it]
      Iteration: 100%|█████████▉| 1742/1750 [43:39<00:12,  1.50s/it]
      Iteration: 100%|█████████▉| 1743/1750 [43:40<00:10,  1.50s/it]
      Iteration: 100%|█████████▉| 1744/1750 [43:42<00:09,  1.50s/it]
      Iteration: 100%|█████████▉| 1745/1750 [43:43<00:07,  1.50s/it]
      Iteration: 100%|█████████▉| 1746/1750 [43:45<00:06,  1.51s/it]
      Iteration: 100%|█████████▉| 1747/1750 [43:47<00:04,  1.51s/it]
      Iteration: 100%|█████████▉| 1748/1750 [43:48<00:03,  1.51s/it]
      Iteration: 100%|█████████▉| 1749/1750 [43:50<00:01,  1.51s/it]
      Iteration: 100%|██████████| 1750/1750 [43:51<00:00,  1.50s/it]
      Epoch: 100%|██████████| 1/1 [43:51<00:00, 2631.58s/it]
      
      
 ### Sample outcomes:
 
    Question: Who did Rollo sign the treaty of Saint-Clair-sur-Epte with?
    Expected Ans: King Charles III
    Predicted Ans: King Charles III of West Francia

    Question: What is the original meaning of the word Norman?
    Expected Ans: Norseman, Viking or  Viking
    Predicted Ans: French normand

    Question: Who ruined Roussel de Bailleul's plans for an independent state?
    Expected Ans: Alexius Komnenos
    Predicted Ans: the Byzantine general Alexius Komnenos

    Question: What was the name of the count of Apulia 
    Expected Ans: Robert Guiscard
    Predicted Ans: Robert Guiscard

    Question: Who did Emma Marry?
    Expected Ans: King Ethelred II
    Predicted Ans: Duke Richard II of Normandy
 
 
 ### Results:
 
    [
     {
       "exact": 51.697127937336816
     },
     {
       "f1": 55.72823471563484
     },
     {
       "total": 11873
     },
     {
       "HasAns_exact": 64.4399460188934
     },
     {
       "HasAns_f1": 72.51371976699234
     },
     {
       "HasAns_total": 5928
     },
     {
       "NoAns_exact": 38.99074852817494
     },
     {
       "NoAns_f1": 38.99074852817494
     },
     {
       "NoAns_total": 5945
     },
     {
       "best_exact": 56.04312305230354
     },
     {
       "best_exact_thresh": -5.20816445350647
     },
     {
       "best_f1": 58.489528046044356
     },
     {
       "best_f1_thresh": -5.112893342971802
     }
      ]


<a id="task2"></a>
 # Task 2
 
 ### Training Logs:
       ======== Epoch 1 / 4 ========
      Training...
        Batch    40  of    241.    Elapsed: 0:00:28.
        Batch    80  of    241.    Elapsed: 0:00:56.
        Batch   120  of    241.    Elapsed: 0:01:24.
        Batch   160  of    241.    Elapsed: 0:01:51.
        Batch   200  of    241.    Elapsed: 0:02:19.
        Batch   240  of    241.    Elapsed: 0:02:47.

        Average training loss: 0.50
        Training epcoh took: 0:02:47

      Running Validation...
        Accuracy: 0.82
        Validation Loss: 0.44
        Validation took: 0:00:06

      ======== Epoch 2 / 4 ========
      Training...
        Batch    40  of    241.    Elapsed: 0:00:28.
        Batch    80  of    241.    Elapsed: 0:00:55.
        Batch   120  of    241.    Elapsed: 0:01:23.
        Batch   160  of    241.    Elapsed: 0:01:51.
        Batch   200  of    241.    Elapsed: 0:02:19.
        Batch   240  of    241.    Elapsed: 0:02:46.

        Average training loss: 0.31
        Training epcoh took: 0:02:47

      Running Validation...
        Accuracy: 0.85
        Validation Loss: 0.42
        Validation took: 0:00:06

      ======== Epoch 3 / 4 ========
      Training...
        Batch    40  of    241.    Elapsed: 0:00:28.
        Batch    80  of    241.    Elapsed: 0:00:55.
        Batch   120  of    241.    Elapsed: 0:01:23.
        Batch   160  of    241.    Elapsed: 0:01:51.
        Batch   200  of    241.    Elapsed: 0:02:19.
        Batch   240  of    241.    Elapsed: 0:02:47.

        Average training loss: 0.20
        Training epcoh took: 0:02:47

      Running Validation...
        Accuracy: 0.84
        Validation Loss: 0.45
        Validation took: 0:00:06

      ======== Epoch 4 / 4 ========
      Training...
        Batch    40  of    241.    Elapsed: 0:00:28.
        Batch    80  of    241.    Elapsed: 0:00:56.
        Batch   120  of    241.    Elapsed: 0:01:23.
        Batch   160  of    241.    Elapsed: 0:01:51.
        Batch   200  of    241.    Elapsed: 0:02:19.
        Batch   240  of    241.    Elapsed: 0:02:47.

        Average training loss: 0.14
        Training epcoh took: 0:02:47

      Running Validation...
        Accuracy: 0.84
        Validation Loss: 0.55
        Validation took: 0:00:06

      Training complete!
      Total training took 0:11:33 (h:mm:ss)
      
 ### Sample predictions
 
 Acceptable = '1', Unacceptable = '0'
 
    Sentence: That professor is feared by all students.
    Expected label: Acceptable
    Predicted label: Acceptable

    Sentence: Mary was given by John the book.
    Expected label: Unacceptable
    Predicted label: Unacceptable

    Sentence: Books were taken from each student and given to Mary by the other.
    Expected label: Unacceptable
    Predicted label: Unacceptable

    Sentence: Max seemed to be trying to begin to love Harriet, and Fred to be trying to begin to love Sue.
    Expected label: Unacceptable
    Predicted label: Acceptable

    Sentence: Whenever Russia has made a major political blunder, the U.S. has too.
    Expected label: Acceptable
    Predicted label: Acceptable

<a id="task3"></a>
# Task 3

### Training logs:
      INFO:filelock:Lock 139867515147216 acquired on /root/.cache/huggingface/transformers/3f12fb71b844fcb7d591fdd4e55027da90d7b5dd6aa5430ad00ec6d76585f26c.58d5dda9f4e9f44e980adb867b66d9e0cbe3e0c05360cefe3cd86f5db4fff042.lock
      Downloading: 100%
      1.60k/1.60k [00:00<00:00, 44.4kB/s]
      INFO:filelock:Lock 139867515147216 released on /root/.cache/huggingface/transformers/3f12fb71b844fcb7d591fdd4e55027da90d7b5dd6aa5430ad00ec6d76585f26c.58d5dda9f4e9f44e980adb867b66d9e0cbe3e0c05360cefe3cd86f5db4fff042.lock
      INFO:filelock:Lock 139867492824912 acquired on /root/.cache/huggingface/transformers/d065edfe6954baf0b989a2063b26eb07e8c4d0b19354b5c74af9a51f5518df6e.6ca4df1a6ec59aa763989ceec10dff41dde19f0f0824b9f5d3fcd35a8abffdb2.lock
      Downloading: 100%
      1.02G/1.02G [00:20<00:00, 46.8MB/s]
      INFO:filelock:Lock 139867492824912 released on /root/.cache/huggingface/transformers/d065edfe6954baf0b989a2063b26eb07e8c4d0b19354b5c74af9a51f5518df6e.6ca4df1a6ec59aa763989ceec10dff41dde19f0f0824b9f5d3fcd35a8abffdb2.lock
      INFO:filelock:Lock 139867501983888 acquired on /root/.cache/huggingface/transformers/0d6fc8b2ef1860c1f8f0baff4b021e3426cc7d11b153f98e563b799603ee2f25.647b4548b6d9ea817e82e7a9231a320231a1c9ea24053cc9e758f3fe68216f05.lock
      Downloading: 100%
      899k/899k [00:00<00:00, 1.62MB/s]
      INFO:filelock:Lock 139867501983888 released on /root/.cache/huggingface/transformers/0d6fc8b2ef1860c1f8f0baff4b021e3426cc7d11b153f98e563b799603ee2f25.647b4548b6d9ea817e82e7a9231a320231a1c9ea24053cc9e758f3fe68216f05.lock
      INFO:filelock:Lock 139867501983888 acquired on /root/.cache/huggingface/transformers/6e75e35f0bdd15870c98387e13b93a8e100237eb33ad99c36277a0562bd6d850.5d12962c5ee615a4c803841266e9c3be9a691a924f72d395d3a6c6c81157788b.lock
      Downloading: 100%
      456k/456k [00:00<00:00, 1.68MB/s]
      INFO:filelock:Lock 139867501983888 released on /root/.cache/huggingface/transformers/6e75e35f0bdd15870c98387e13b93a8e100237eb33ad99c36277a0562bd6d850.5d12962c5ee615a4c803841266e9c3be9a691a924f72d395d3a6c6c81157788b.lock
      INFO:filelock:Lock 139867500691664 acquired on /root/.cache/huggingface/transformers/d94f53c8851dcda40774f97280e634b94b721a58e71bcc152b5f51d0d49a046a.fc9576039592f026ad76a1c231b89aee8668488c671dfbe6616bab2ed298d730.lock
      Downloading: 100%
      1.36M/1.36M [00:00<00:00, 4.84MB/s]
      INFO:filelock:Lock 139867500691664 released on /root/.cache/huggingface/transformers/d94f53c8851dcda40774f97280e634b94b721a58e71bcc152b5f51d0d49a046a.fc9576039592f026ad76a1c231b89aee8668488c671dfbe6616bab2ed298d730.lock
      INFO:filelock:Lock 139867500691664 acquired on /root/.cache/huggingface/transformers/1abf196c889c24daca2909359ca2090e5fcbfa21a9ea36d763f70adbafb500d7.67d01b18f2079bd75eac0b2f2e7235768c7f26bd728e7a855a1c5acae01a91a8.lock
      Downloading: 100%
      26.0/26.0 [00:00<00:00, 655B/s]
      INFO:filelock:Lock 139867500691664 released on /root/.cache/huggingface/transformers/1abf196c889c24daca2909359ca2090e5fcbfa21a9ea36d763f70adbafb500d7.67d01b18f2079bd75eac0b2f2e7235768c7f26bd728e7a855a1c5acae01a91a8.lock
      INFO:simpletransformers.seq2seq.seq2seq_utils: Creating features from dataset file at cache_dir/
      100%
      5000/5000 [00:02<00:00, 2752.58it/s]
      INFO:simpletransformers.seq2seq.seq2seq_model: Training started
      Epoch 1 of 1: 100%
      1/1 [13:32<00:00, 812.69s/it]
      wandb: You can find your API key in your browser here: https://wandb.ai/authorize
      wandb: Paste an API key from your profile and hit enter: ··········
      huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
      To disable this warning, you can either:
         - Avoid using `tokenizers` before the fork if possible
         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
      huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
      To disable this warning, you can either:
         - Avoid using `tokenizers` before the fork if possible
         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
      2021-08-11 09:33:20.833793: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library libcudart.so.11.0
      wandb: W&B syncing is set to `offline` in this directory.  Run `wandb online` or set WANDB_MODE=online to enable cloud syncing.
      Epochs 0/1. Running Loss: 0.5364: 100%
      625/625 [09:41<00:00, 1.07it/s]
      INFO:simpletransformers.seq2seq.seq2seq_model:Saving model into outputs/checkpoint-625-epoch-1
      INFO:simpletransformers.seq2seq.seq2seq_utils: Creating features from dataset file at cache_dir/
      huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
      To disable this warning, you can either:
         - Avoid using `tokenizers` before the fork if possible
         - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
      100%
      500/500 [00:01<00:00, 1.21s/it]
      INFO:simpletransformers.seq2seq.seq2seq_model:{'eval_loss': 0.5591612346470356}
      INFO:simpletransformers.seq2seq.seq2seq_model:Saving model into outputs/best_model
      INFO:simpletransformers.seq2seq.seq2seq_model:Saving model into outputs/
      INFO:simpletransformers.seq2seq.seq2seq_model: Training of facebook/bart-large model complete. Saved to outputs/.
      
 ### Sample outcomes:
 
