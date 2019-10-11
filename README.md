#title

##Overview
In MicroNet Challenge, we consider “WikiText-103 Language Modeling” task. This task required number of parameters and math operations of the learned model are minimum. And, the need condition is the perplexity of model below 35 on the test set. With this goal, we proposed an approach based on QRNN model (Quasi-Recurrent Neural Networks). By turing parameter, we reduced the number of parameters and sure the perplexity below 35. Parameter tuning focus on 3 parameters: sequence length, embedding size and number of hidden units per layer. After changing, number of parameters of our model reduce approx 32% when compared to the default model.

##Method
###Base Model

Our approach based on the Quasi-Recurrent Neural Networks - an approach to neural sequence modeling that alternates convolutional layers, which apply in parallel across timesteps, and a minimalist recurrent pooling function that applies in parallel across channels (link paper).

In “An Analysis of Neural Language Modeling at Multiple Scales”, QRNN was used for wiki-103 dataset, with parameters setting:
+ Number of epoch (--epochs): 14 
+ Number of layer (--nlayers): 4 
+ Size of word embeddings (--emsize): 400 
+ Number of hidden units per layer (--nhid): 2500 
+ Alpha L2 regularization on RNN activation (--alpha): 0 
+ Beta slowness regularization applied on RNN activiation (--beta): 0 
+ Dropout to remove words from embedding layer (--dropoute): 0 
+ Dropout for rnn layers (--dropouth): 0.1 
+ Dropout for input embedding layers (--dropouti): 0.1 
+ Dropout applied to layers (--dropout): 0.1 
+ Amount of weight dropout to apply to the RNN hidden to hidden matrix (--wdrop): 0 
+ Weight decay applied to all weights (--wdecay): 0 
+ Sequence length (--bptt): 140 
+ Batch size (--batch_size): 60 
+ Optimizer to use (--optimizer): adam 
+ Learning rate (--lr): 1e-3

With this setting, the result of QRNN when run on wiki-103 dataset:
Total parameters: 153,886,638
Test perplexity: 32.58 

###Parameter Tuning
We focus on reducing the number of parameters of model follow 2 ways: Sequence length (--bptt) and Embedding size and number of hidden units per layer(--emsize, --nhid).

####Sequence length in training 
In QRNN model, there is a parameter is “--bptt”. It is sequence length. In training, all training data was connected and created a long sequence. Sequence length will divide the sequence for sub-sequences with length is the value of “-bptt”. We tried to change this value to analysis effective for ppl. The recommended value is 140. The default value is 70. The figure shows the perplexity with the change of sequence length. 

![Perplexity](https://github.com/binhdt95/Jaist-MicroNet-Challenge/image/chart.png)

####Embedding size and number of hidden units per layer
With two parameters, we expect change value to reduce the number of parameters of model and keep value of test ppl < 35. The default model use the embedding size is 400 and the number of hidden units per layer is 2500. With the setting, the default had 153M parameter and the perplexity reached 32.58. Our idea is to reduce the number of parameters. This make increase perplexity. So that, we try to balance for ppl not over 35. To reduce parameter, embedding size decreased to 300. This number is a popular choice for embedding size in deep learning models. With that, the number of hidden units per layer also change for fit. The table shows some our experiments with this changing.

![Result] (https://github.com/binhdt95/Jaist-MicroNet-Challenge/image/experiments.png)

With some experiments, we chose --emsize 300 --nhid 1800 --bptt 300 is the best state. 

![best](https://github.com/binhdt95/Jaist-MicroNet-Challenge/image/best-state.png)

##System Configuration
###Software requirements (codebase)
+ Python 3
+ Pytorch: 0.4
+ pynvrtc (NVIDIA's Python Bindings to NVRTC) (pip install git+git://github.com/NVIDIA/pynvrtc/commit/6417a2896ff8a99f2c4d4195de657671a77c89a0)
###How to run
+ Install pytorch 0.4, pynvrtc.
+ `python -u main.py --epochs 20 --nlayers 4 --emsize 300 --nhid 1800 --alpha 0 --beta 0 --dropoute 0 --dropouth 0.1 --dropouti 0.1 --dropout 0.1 --wdrop 0 --wdecay 0 --bptt 300 --batch_size 40 --optimizer adam --lr 1e-3 --data data/wikitext-103 --save WT103.12hr.QRNN.pt --when 12 --model QRNN`



