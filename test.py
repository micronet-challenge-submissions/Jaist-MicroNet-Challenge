from model import RNNModel
import math
import counting
import torch
import time
import numpy as np
import torch

from utils import batchify, get_batch, repackage_hidden
import collections

def read_model(model):
    assert isinstance(model,RNNModel)
    assert model.rnn_type  == 'QRNN'
    ops = []

    encoder, rnns, decoder = model.encoder, model.rnns, model.decoder
    ops+=[
        ('embedding',counting.Embedding(input_size=encoder.weight.size()[0], n_channels=encoder.weight.size()[1]))
    ]

    ops+=[('block_qrnn',[('qrnn_%d'%i,counting.QRNN(input_size=l.input_size, 
                                            hidden_size=l.hidden_size, 
                                            window=l.window, output_gate=l.output_gate))
                        for i,l in enumerate(rnns)])
        ]

    ops+=[
        ('block_decoder',[
            # separate weight and bias for convenient counting with tied weights
            ('decoder_weight',counting.FullyConnected(kernel_shape=(decoder.weight.size()[1], decoder.weight.size()[0]), 
                                use_bias=False, tied_weights=model.tie_weights)),
            ('decoder_bias',counting.FullyConnected(kernel_shape=(0, decoder.weight.size()[0]), 
                                use_bias=getattr(decoder,'bias', None) is not None)),
            ('max',counting.GlobalMax(input_size=1, n_channels=decoder.weight.size()[0]))
            ])
    ]

    return ops

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if model.rnn_type == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

args=(collections.namedtuple('Args',['bptt','cuda']))(bptt=140,cuda=True)

# load model
model_load('WT103.12hr.QRNN.pt')

# load test data: read vocab, process test text
corpus = torch.load('corpus-wikitext-103.vocab-only.data')
test_data = corpus.tokenize('data/wikitext-103/test.txt')
test_data = batchify(test_data, 1, args)
# Run on test data.
test_loss = evaluate(test_data, 1)
print('=' * 89)
print('Test ppl {:8.2f} '.format(math.exp(test_loss)))
print('=' * 89)

# read model ops
ops = read_model(model)
# print model MFLOPS and #Parameters
counter = counting.MicroNetCounter(ops, add_bits_base=32, mul_bits_base=32)
INPUT_BITS = 16
ACCUMULATOR_BITS = 32
PARAMETER_BITS = INPUT_BITS
SUMMARIZE_BLOCKS = True
counter.print_summary(0, PARAMETER_BITS, ACCUMULATOR_BITS, INPUT_BITS, summarize_blocks=SUMMARIZE_BLOCKS)
