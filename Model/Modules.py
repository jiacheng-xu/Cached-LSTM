__author__ = 'jcxu'

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


def param_init_gru(options, params, prefix='gru', nin=None, dim=None):
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
    params[_p(prefix, 'V')] = U

    params[_p(prefix, 'b')] = numpy.zeros((3 * dim,)).astype('float32')

    return params


def gru(tparams, state_below, options, prefix='gru', mask=None,
             out_dim=None):
    if out_dim is None:
        out_dim = options['dim_hidden']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # dim = tparams[_p(prefix, 'Ux')].shape[1]

    # only one sample case
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)
    mask = mask[:-1]
    # utility function to slice a tensor
    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    # state_below is input embedding
    state_below_ = tensor.dot(state_below[:-1], tparams[_p(prefix, 'U')]) + \
                   tensor.dot(state_below[1:], tparams[_p(prefix, 'V')]) +\
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

        # compute the hidden state proposal
        # preactx = tensor.dot(h_, Ux)
        # preactx = preactx * r
        # preactx = preactx + xx_
        # hidden state proposal
        # h = tensor.tanh(preactx)

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
                                n_steps=nsteps-1)
                                # profile=profile,
                                # strict=True)
    # rval = [rval]
    return rval  # GRU layer
"""
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
    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # recurrent transformation weights for hidden state proposal
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux

    return params


def gru_layer(tparams, state_below, options, prefix='gru', mask=None,
              **kwargs):
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
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval
"""

def param_init_blstm(options, params, prefix='blstm', in_dim=None, out_dim=None):
    """
    Use weights between forward and backward.
    """
    if in_dim is None:
        in_dim = options['dim_proj']
    if out_dim is None:
        out_dim = options['dim_proj']

    Wf = numpy.concatenate([ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim)], axis=1)
    params[_p(prefix, 'Wf')] = Wf
    Uf = numpy.concatenate([ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Uf')] = Uf
    bf = numpy.zeros((4 * out_dim,))
    params[_p(prefix, 'bf')] = bf.astype(config.floatX)

    Wb = numpy.concatenate([ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim)], axis=1)
    params[_p(prefix, 'Wb')] = Wb
    Ub = numpy.concatenate([ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Ub')] = Ub
    bb = numpy.zeros((4 * out_dim,))
    params[_p(prefix, 'bb')] = bb.astype(config.floatX)

    Vf = numpy.concatenate([ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vf')] = Vf
    Vb = numpy.concatenate([ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vb')] = Vb
    bo = numpy.zeros((out_dim,)).astype(config.floatX)
    params[_p(prefix, 'bo')] = bo
    return params


def blstm(tparams, state_below, options, prefix='blstm', mask=None, in_dim=None, out_dim=None):
    """
    Bidirectional lstm, get the whole h layer
    :param tparams:
    :param state_below: x
    :param options:
    :param prefix:
    :param mask:
    :return: array nsamples, batch_szie, ndim*2
    """
    if out_dim is None:
        out_dim = options['dim_proj']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'Wf')]) + tparams[_p(prefix, 'bf')]

    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wb')]) + tparams[_p(prefix, 'bb')]

    def _step(m_, x_, h_, c_, U):
        preact = tensor.dot(h_, U)
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, out_dim))
        c = tensor.tanh(_slice(preact, 3, out_dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h, c

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below_],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim)],
                                non_sequences=[tparams[_p(prefix, 'Uf')]],
                                name=_p(prefix, '_flayers'),
                                n_steps=nsteps)

    bval, updates = theano.scan(_step,
                                sequences=[mask, state_belowx],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim)],
                                non_sequences=[tparams[_p(prefix, 'Ub')]],
                                name=_p(prefix, '_blayers'),
                                go_backwards=True,
                                n_steps=nsteps)
    #####
    #
    rt_fwd = rval[0]  # h of forward step
    rt_bwd = bval[0][::-1, :, :]  # h of backward step, and reverse in the axis=0

    ret_h = tensor.dot(rt_fwd, tparams[_p(prefix, 'Vf')]) + tensor.dot(rt_bwd, tparams[_p(prefix, 'Vb')]) + tparams[
        _p(prefix, 'bo')]

    # Like 0,1,2    3,4,5
    #      3,4,5    0,1,2
    #
    ####
    # 97,16,128  *  2   ==> 97,16,256
    # sum
    # rt = concatenate([rval[0], bval[0]], axis=rval[0].ndim - 1)  # 97,16, 256
    # end
    # rt = concatenate([rval[0][-1], bval[0][-1]], axis=rval[0][-1].ndim - 1)  # 16, 256

    # rt = [rval[0], bval[0]]
    return ret_h


def param_init_lstm(options, params, prefix='lstm', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['dim_proj']
    if out_dim is None:
        out_dim = options['dim_proj']

    W = numpy.concatenate([ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * out_dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    return params


def lstm(tparams, state_below, options, prefix='lstm', mask=None, in_dim=None, out_dim=None):
    if out_dim is None:
        out_dim = options['dim_proj']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
        f = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, out_dim))
        c = tensor.tanh(_slice(preact, 3, out_dim))

        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = out_dim
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


def param_init_lstm_full(options, params, prefix='lstm_full', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['dim_proj']
    if out_dim is None:
        out_dim = options['dim_proj']

    W = numpy.concatenate([ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim),
                           ortho_weight(in_dim, out_dim)], axis=1)
    params[_p(prefix, 'W')] = W

    U = numpy.concatenate([ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'U')] = U

    b = numpy.zeros((4 * out_dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    Vi = ortho_weight(out_dim, out_dim)
    params[_p(prefix, 'Vi')] = Vi
    Vo = ortho_weight(out_dim, out_dim)
    params[_p(prefix, 'Vo')] = Vo
    Vf = ortho_weight(out_dim, out_dim)
    params[_p(prefix, 'Vf')] = Vf
    return params


def lstm_full(tparams, state_below, options, prefix='lstm_full', mask=None, in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['dim_proj']
    if out_dim is None:
        out_dim = options['dim_proj']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        i = tensor.nnet.sigmoid(_slice(preact, 0, out_dim) + tensor.dot(c_, tparams[_p(prefix, 'Vi')]))
        f = tensor.nnet.sigmoid(_slice(preact, 1, out_dim) + tensor.dot(c_, tparams[_p(prefix, 'Vf')]))

        c = tensor.tanh(_slice(preact, 3, out_dim))
        c = f * c_ + i * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        o = tensor.nnet.sigmoid(_slice(preact, 2, out_dim) + tensor.dot(c, tparams[_p(prefix, 'Vo')]))
        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = out_dim
    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]


def param_init_weight(options, params, prefix='weight',
                      dim=None):
    if dim is None:
        dim = options['dim_proj']

    params[_p(prefix, 'L')] = ortho_weight(dim)
    params[_p(prefix, 'R')] = ortho_weight(dim)
    t = numpy.zeros((dim,))
    params[_p(prefix, 't')] = t.astype(config.floatX)
    return params


def weight(tparams, state_below, options, state_below_option=None, prefix='weight', mask=None, double=False):
    # TODO
    if state_below_option is None:
        input_state = tensor.dot(state_below[:-1], tparams[_p(prefix, 'L')]) \
                      + tensor.dot(state_below[1:], tparams[_p(prefix, 'R')]) + \
                      tparams[_p(prefix, 't')]
    else:
        input_state = tensor.dot(state_below, tparams[_p(prefix, 'L')]) + tensor.dot(state_below_option,
                                                                                     tparams[_p(prefix, 'R')]) + \
                      tparams[_p(prefix, 't')]
    input_state = tensor.tanh(input_state)
    return input_state


def param_init_attention(options, params, prefix='attention', dim=None):
    if dim is None:
        dim = options['dim_hidden']

    W = ortho_weight(dim)
    params[_p(prefix, 'W')] = W
    b = numpy.zeros((dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    U = numpy.random.uniform(low=-.1, high=.1, size=(dim,)).astype(config.floatX)
    params[_p(prefix, 'U')] = U

    return params


def attention(tparams, state_below, options, prefix='attention', mask=None):
    # pctx = 97,16,256  256*256  + 256, = 97,16,256
    pctx = tensor.dot(state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    pctx_ = tensor.tanh(pctx)

    # alpha = 97,16,256 * 256,  = 97,16
    alpha = tensor.dot(pctx_, tparams[_p(prefix, 'U')])
    alpha = tensor.exp(alpha)
    alpha = alpha * mask
    alpha = alpha / theano.tensor.sum(alpha, axis=0, keepdims=True)
    # alpha.sum(axis=0)
    # h = emb * alpha[:, :, None]
    # h = tensor.dot(state_below,alpha)
    # h = state_below * alpha[:, :, None]
    # alpha
    h = alpha[:, :, None] * state_below
    # proj = (h * mask[:, :, None]).sum(axis=0)
    # proj = proj / mask.sum(axis=0)[:, None]
    # proj = tensor.tanh(tensor.dot(proj, tparams[_p(prefix, 'O')]))
    return h


def param_init_gate(options, params, prefix='gate', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['dim_hidden']
    if out_dim is None:
        out_dim = options['dim_hidden']

    # T = ortho_weight(in_dim, out_dim)
    # params[_p(prefix, 'T')] = T

    Wl = ortho_weight(out_dim)
    params[_p(prefix, 'Wl')] = Wl
    Wr = ortho_weight(out_dim)
    params[_p(prefix, 'Wr')] = Wr
    params[_p(prefix, 'Wb')] = numpy.zeros((out_dim,)).astype(config.floatX)

    Gl = ortho_weight(out_dim, 3)
    params[_p(prefix, 'Gl')] = Gl
    Gr = ortho_weight(out_dim, 3)
    params[_p(prefix, 'Gr')] = Gr
    params[_p(prefix, 'Gb')] = numpy.zeros((3,)).astype(config.floatX)

    params[_p(prefix, 'V')] = numpy.random.uniform(low=-.1, high=.1, size=(3, 3)).astype(config.floatX)
    params[_p(prefix, 'b')] = numpy.zeros((3,)).astype(config.floatX)
    return params


def gate(tparams, state_below, options, prefix='gate', dim=None, state_below_option=None):
    """
    Gate Architecture
    :param tparams:
    :param state_below: Default input
    :param options:
    :param prefix:
    :param dim:
    :param state_below_option: Optional input.
    :return:
    """
    if state_below_option is None:
        state_l = state_below[:-1]
        state_r = state_below[1:]
    else:
        state_l = state_below
        state_r = state_below_option

    # state_below = tensor.dot(state_below, tparams[_p(prefix, 'T')])
    h_hat = tensor.dot(state_l, tparams[_p(prefix, 'Wl')]) + tensor.dot(state_r, tparams[_p(prefix, 'Wr')]) \
            + tparams[_p(prefix, 'Wb')]
    h_hat = tensor.tanh(h_hat)  # 96,16,100

    g = tensor.dot(state_l, tparams[_p(prefix, 'Gl')]) + tensor.dot(state_r, tparams[_p(prefix, 'Gr')]) \
        + tparams[_p(prefix, 'Gb')]

    # g ==> 96,16,3

    # input 96, 16, 3
    # output 96, 16, 3 (softmaxed)
    def soft(g):
        # mask 96,16   g 96,16,3
        return tensor.nnet.softmax(tensor.dot(g, tparams[_p(prefix, 'V')]) + tparams[_p(prefix, 'b')])

    rval, updates = theano.scan(soft,
                                sequences=[g],
                                outputs_info=[None],
                                name=_p(prefix, 'softmax'))

    # omega = tensor.nnet.softmax(tensor.dot(g, tparams[_p(prefix, 'V')]) + tparams[_p(prefix, 'b')])
    omega = rval
    # 96, 16, 3
    wc = (omega[:, :, 0])[:, :, None]
    wl = (omega[:, :, 1])[:, :, None]
    wr = (omega[:, :, 2])[:, :, None]
    # wc wl wr 96,16,1

    h_out = wc * h_hat + wl * state_l + wr * state_r
    return h_out
