
from collections import OrderedDict
import theano
import theano.tensor as tensor
import numpy
import cPickle as pkl
import sys
import time
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import config
from Util import *
proile=True
# gru
# gru_conv

def _p(pp, name):
    """
    make prefix-appended name
    :param pp: prefix
    :param name: name
    :return: pp_name
    """
    return '%s_%s' % (pp, name)


def numpy_floatX(data):
    return numpy.asarray(data, dtype=config.floatX)

def param_init_conv(options, params, prefix='conv', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']
    # U for h^{t}_{i-1, j-1} left
    # U = numpy.concatenate([ortho_weight(nin, dim),
    #                        ortho_weight(nin, dim),
    #                        ortho_weight(nin, dim)], axis=1)
    U = ortho_weight(nin, dim)
    params[_p(prefix, 'U')] = U

    # V for h^{t}_{u-1, j} right
    # V = numpy.concatenate([ortho_weight(nin, dim),
    #                        ortho_weight(nin, dim),
    #                        ortho_weight(nin, dim)], axis=1)
    V = ortho_weight(nin, dim)
    params[_p(prefix, 'V')] = V

    params[_p(prefix, 'b')] = numpy.zeros((dim,)).astype('float32')

    return params
def param_init_gru_conv(options, params, prefix='gru_conv', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases

    # W for h^{t-1}
    W = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'W')] = W
    # U for h^{t}_{i-1, j-1} left
    U = numpy.concatenate([ortho_weight(nin, dim),
                           ortho_weight(nin, dim),
                           ortho_weight(nin, dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # V for h^{t}_{u-1, j} right
    V = numpy.concatenate([ortho_weight(nin, dim),
                           ortho_weight(nin, dim),
                           ortho_weight(nin, dim)], axis=1)
    params[_p(prefix, 'V')] = V

    params[_p(prefix, 'b')] = numpy.zeros((3 * dim,)).astype('float32')

    return params

def gru_conv(tparams, left_state, right_state, options, prefix, mask,
             out_dim):

    # nsteps = left_state.shape[0]
    if left_state.ndim == 3:
        n_samples = left_state.shape[1]
    else:
        n_samples = 1

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is input embedding
    state_below_ = tensor.dot(left_state, tparams[_p(prefix, 'U')]) + \
                   tensor.dot(right_state, tparams[_p(prefix, 'V')]) +\
                   tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    # state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
    # tparams[_p(prefix, 'bx')]


    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, h_, W):
        """
        m_: mask nsteps, batch
        x_: U*h_left+V*h_right+b = 97,16,50 * 50,300 = 97,16,300
        h_: hidden_{t-1}
        W :
        Assume batch size=16, hidden_unit=100, input_unit=50
        """
        preact = tensor.dot(h_, W)  # 16, 100 dot 100, 300 = 16*300
        # preact_gate = _slice(preact, 3, dim)
        # preact += x_ # 16, 300
        r = tensor.nnet.sigmoid(_slice(preact, 0, out_dim) + _slice(x_, 0, out_dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, out_dim) + _slice(x_, 1, out_dim))

        # reset and update gates
        h_hat = tensor.tanh(_slice(preact, 2, out_dim) * r + _slice(x_, 2, out_dim))

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h_hat
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h


    # prepare scan arguments
    seqs = [mask, state_below_]
    init_states = [tensor.alloc(0., n_samples, out_dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'W')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                # n_steps=nsteps
                                )
                                # profile=profile,
                                # strict=True)
    # rval = [rval]
    return rval  # GRU layer


def gru_conv_naive(tparams, state_below, options, prefix, mask,
             out_dim):

    left_state = state_below[:-1]
    right_state = state_below[1:]

    # nsteps = left_state.shape[0]
    if left_state.ndim == 3:
        n_samples = left_state.shape[1]
    else:
        n_samples = 1

    # dim = tparams[_p(prefix, 'Ux')].shape[1]

    # only one sample case
    if mask is None:
        mask = tensor.alloc(1., left_state.shape[0], 1)
    # mask = mask[:-1]
    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is input embedding
    state_below_ = tensor.dot(left_state, tparams[_p(prefix, 'U')]) + \
                   tensor.dot(right_state, tparams[_p(prefix, 'V')]) +\
                   tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    # state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
    # tparams[_p(prefix, 'bx')]


    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, h_, W):
        """
        m_: mask nsteps, batch
        x_: U*h_left+V*h_right+b = 97,16,50 * 50,300 = 97,16,300
        h_: hidden_{t-1}
        W :
        Assume batch size=16, hidden_unit=100, input_unit=50
        """
        preact = tensor.dot(h_, W)  # 16, 100 dot 100, 300 = 16*300
        # preact_gate = _slice(preact, 3, dim)
        # preact += x_ # 16, 300
        r = tensor.nnet.sigmoid(_slice(preact, 0, out_dim) + _slice(x_, 0, out_dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, out_dim) + _slice(x_, 1, out_dim))

        # reset and update gates
        h_hat = tensor.tanh(_slice(preact, 2, out_dim) * r + _slice(x_, 2, out_dim))

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h_hat
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h


    # prepare scan arguments
    seqs = [mask, state_below_]
    init_states = [tensor.alloc(0., n_samples, out_dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'W')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                # n_steps=nsteps
                                )
                                # profile=profile,
                                # strict=True)
    # rval = [rval]
    return rval  # GRU layer

def conv(tparams, left_state, right_state, options, prefix, mask,
             out_dim):
    state_below_ = tensor.dot(left_state, tparams[_p(prefix, 'U')]) + \
                   tensor.dot(right_state, tparams[_p(prefix, 'V')]) +\
                   tparams[_p(prefix, 'b')]
    return state_below_

def conv_naive(tparams, state_below, options, prefix, mask,
             out_dim):
    left_state = state_below[:-1]
    right_state = state_below[1:]
    state_below_ = tensor.dot(left_state, tparams[_p(prefix, 'U')]) + \
                   tensor.dot(right_state, tparams[_p(prefix, 'V')]) +\
                   tparams[_p(prefix, 'b')]
    return state_below_

def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
    if nin is None:
        nin = options['dim_proj']
    if dim is None:
        dim = options['dim_proj']

    # embedding to gates transformation weights, biases
    W = numpy.concatenate([ortho_weight(nin, dim),
                           ortho_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    # recurrent transformation weights for gates
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    # embedding to hidden state proposal weights, biases
    Wx = ortho_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params

#tparams, state_below, options, prefix='gru_conv', mask=None,
             # out_dim=None
def gru(tparams, state_below, options, prefix='gru', mask=None,
              out_dim=None):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is the input word embeddings
    # input to the gates, concatenated
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + \
                   tparams[_p(prefix, 'b')]
    # input to compute the hidden state proposal
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) + \
                   tparams[_p(prefix, 'bx')]

    # step function to be used by scan
    # arguments    | sequences |outputs-info| non-seqs
    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        # reset and update gates
        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        # compute the hidden state proposal
        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        # hidden state proposal
        h = tensor.tanh(preactx)

        # leaky integrate and obtain next hidden state
        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    # prepare scan arguments
    seqs = [mask, state_below_, state_belowx]
    init_states = [tensor.alloc(0., n_samples, dim)]
    _step = _step_slice
    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Ux')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=init_states,
                                non_sequences=shared_vars,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                # profile=profile,
                                strict=True)
    # rval = [rval]
    return rval