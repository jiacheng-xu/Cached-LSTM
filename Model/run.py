__author__ = 'jcxu'
import sys
from collections import OrderedDict
import theano
import theano.tensor as tensor
import numpy
from template import *
import os

decay = float(sys.argv[1])
drop = float(sys.argv[2])
dataname= sys.argv[3]
name = 'gru_5_nodes'


data=dataname+'_50_100000_glove.twitter.27B'
# dataname='Yelp2013'

decay_list = [1e-5, 1e-4]
drop_list = [0,0.3,0.5]

hidden = 50


train_err, valid_err, test_err = train(
    dataname=dataname,
    dataset=data,
    dim_proj=50,
    dim_hidden=hidden,
    max_epochs = 40,
    use_dropout=True,
    noise_std=drop,
    patience=15,
    optimizer=adagrad,
    decay_c=decay,
    disp_frq=1000,
    valid_freq=-1,
    batch_size=16,
    lrate=0.2,
    lrate_embed=0.5,
    end=True
)
if os.path.isfile((name + '.md')) is False:
    f = open((name + '.md'), 'w')
    f.close()

f = open((name + '.md'), 'a')
f.write('|' + str(decay) +'|'+data+'|' + str(drop) + '|' + str(
    1. - train_err)
        # +'|' + str(1. - train4valid)
        + '|' + str(1. - valid_err) + '|' + str(1. - test_err) + '|\n')
f.close()
