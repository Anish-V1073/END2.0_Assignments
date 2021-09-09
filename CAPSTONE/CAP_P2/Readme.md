# END2.0 CAPSTONE Part 2

## About CAPSTONE

In this project we are implementing a QuestionAnswering model on Pytorch. We have collected the data regarding the Pytorch from Stackoverflow, Pytorch discussion forum, Youtube(Subtitles of Pytorch related videos) and Pytorch documentation.

The Goal of this CAPSTONE Project is to build a closed domain(only Pytorch related) QuestionAnswering system. Once the data is collected from the different forums on Pytorch, the data is formated into  JSON format.
      
      {
            "X":Question
            "Y":Answer
            "Z":Answer document
      }
Below are some of the sample examples in a dataset:

      {
        "X": "What module implements quantized versions of the nn layers?",
        "Y": "torch.nn.quantized.dynamic",
        "Z": "torch.nn.intrinsic.quantized This module implements the quantized implementations of fused operations\nlike conv + relu. torch.nn.qat This module implements versions of the key nn modules Conv2d() and\nLinear() which run in FP32 but with rounding applied to simulate the\neffect of INT8 quantization. torch.nn.quantized This module implements the quantized versions of the nn layers such as\n~`torch.nn.Conv2d` and torch.nn.ReLU. torch.nn.quantized.dynamic"
      },
      {
        "X": "What is one of the quantized versions of nn layers?",
        "Y": "RNNCell",
        "Z": "torch.nn.intrinsic This module implements the combined (fused) modules conv + relu which can\nthen be quantized. torch.nn.intrinsic.qat This module implements the versions of those fused operations needed for\nquantization aware training. torch.nn.intrinsic.quantized This module implements the quantized implementations of fused operations\nlike conv + relu. torch.nn.qat This module implements versions of the key nn modules Conv2d() and\nLinear() which run in FP32 but with rounding applied to simulate the\neffect of INT8 quantization. torch.nn.quantized This module implements the quantized versions of the nn layers such as\n~`torch.nn.Conv2d` and torch.nn.ReLU. torch.nn.quantized.dynamic Dynamically quantized Linear, LSTM,\nLSTMCell, GRUCell, and\nRNNCell."
      },
      {
        "X": "What is quant_min in torch.fake_quantize_per_tensor_affine?",
        "Y": "the lower bound of quantized domain",
        "Z": "torch.fake_quantize_per_tensor_affine Returns a new tensor with the data in input fake quantized using scale,\nzero_point, quant_min and quant_max. input (Tensor) \u2013 the input value(s), in torch.float32. scale (double) \u2013 quantization scale zero_point (int64) \u2013 quantization zero_point quant_min (int64) \u2013 lower bound of the quantized domain quant_max (int64) \u2013 upper bound of the quantized domain A newly fake_quantized tensor Tensor Example:"
      },



To Achieve this task we need three main components:
* A pre-trained generator model like BERT
* A pre-trained retriever model like DPR
* An Indexed KB of text Documents(Answer documents, JSON file)
      


