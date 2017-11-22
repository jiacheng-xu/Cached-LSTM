# from module import *
import theano.tensor as tensor
import numpy
from util import glorot_uniform, ortho_weight
import theano
from theano import config


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


def param_init_forget(options, params):
    pipe_list = []
    start = 0.
    end = 1.
    step = (end - start) / (options['pipe_num'] + 1)
    for i in xrange(options['pipe_num']):
        pipe_list.append((i + 1) * step + start)
    print('Pipe list:', pipe_list)
    pipe_numpy = numpy.array(pipe_list)

    forget_matrix = numpy.repeat(pipe_numpy, options['edim']).astype(dtype=config.floatX)

    # forget_bias = numpy.repeat(pipe_bias, options['edim']).astype(dtype=config.floatX)
    print('Forget Matrix Shape:', forget_matrix.shape)
    options['forget'] = forget_matrix
    return options


def param_init_dynamic(options, params):
    bias_list = []
    start = 0.
    end = 1.
    step = (end - start) / (options['pipe_num'])
    for i in xrange(options['pipe_num']):
        bias_list.append((i) * step + start)
    print('Pipe list:', bias_list)
    pipe_numpy = numpy.array(bias_list)

    forget_bias = numpy.repeat(pipe_numpy, options['edim']).astype(dtype=config.floatX)
    options['step'] = step
    options['forget_bias'] = forget_bias

    return options


def param_init_multi_lstm_s(options, params, prefix='multi_lstm_s', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['edim']

    list_w = []
    for i in xrange(options['pipe_num']):
        if options['with_gate'] == True:
            list_w.append(numpy.concatenate([glorot_uniform(in_dim, out_dim),
                                             glorot_uniform(in_dim, out_dim),
                                             glorot_uniform(in_dim, out_dim, 4.)], axis=1))
        else:
            list_w.append(numpy.concatenate([glorot_uniform(in_dim, out_dim),
                                             glorot_uniform(in_dim, out_dim, 4.)], axis=1))
    params[_p(prefix, 'W')] = numpy.concatenate(list_w, axis=1)

    list_U = []
    for i in xrange(options['pipe_num']):
        if options['with_gate'] == True:
            list_U.append(numpy.concatenate([ortho_weight(options['pipe_num'] * out_dim, out_dim),
                                             ortho_weight(options['pipe_num'] * out_dim, out_dim),
                                             ortho_weight(options['pipe_num'] * out_dim, out_dim)], axis=1))
        else:
            list_U.append(numpy.concatenate([ortho_weight(options['pipe_num'] * out_dim, out_dim),
                                             ortho_weight(options['pipe_num'] * out_dim, out_dim)], axis=1))
    U = numpy.concatenate(list_U, axis=1)
    params[_p(prefix, 'U')] = U

    if options['with_gate'] == True:
        b = numpy.zeros((3 * options['pipe_num'] * out_dim,)).astype(config.floatX)
        params[_p(prefix, 'b')] = b
    else:
        params[_p(prefix, 'b')] = numpy.zeros((2 * options['pipe_num'] * out_dim,)).astype(config.floatX)

    # print('Wshape %s Ushape %s '%(W.shape,U.shape))
    # print(b.shape)
    return params


def param_init_multi_blstm_s(options, params, prefix='multi_blstm_s', in_dim=None, out_dim=None):
    """
    Use weights between forward and backward.
    """
    if in_dim is None:
        in_dim = options['dim_proj']
    if out_dim is None:
        out_dim = options['dim_proj']
    params = param_init_multi_lstm_s(options, params, prefix=prefix + '_f', in_dim=in_dim, out_dim=out_dim)
    params = param_init_multi_lstm_s(options, params, prefix=prefix + '_b', in_dim=in_dim, out_dim=out_dim)

    Vf = numpy.concatenate([glorot_uniform(options['pipe_num'] * out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vf')] = Vf
    Vb = numpy.concatenate([glorot_uniform(options['pipe_num'] * out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vb')] = Vb
    bo = numpy.zeros((out_dim,)).astype(config.floatX)
    params[_p(prefix, 'bo')] = bo

    return params


def multi_lstm_s(tparams, state_below, options, prefix='multi_lstm_s', mask=None, in_dim=None, out_dim=None):
    if out_dim is None:
        out_dim = options['edim']
    pipe_num = options['pipe_num']
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim * pipe_num:(n + 1) * dim * pipe_num]
        return _x[:, n * dim * pipe_num:(n + 1) * dim * pipe_num]

    if options['with_gate'] == True:

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
            preact += x_

            io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim)) * tparams['forget']  # batch_size, pipe_num*edim
            o = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
            c = tensor.tanh(_slice(preact, 2, out_dim))

            c = (io) * c_ + (1 - io) * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c
    else:

        def _step(m_, x_, h_, c_):
            preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
            preact += x_

            io = tparams['forget']  # batch_size, pipe_num*edim
            o = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
            c = tensor.tanh(_slice(preact, 1, out_dim))

            c = (io) * c_ + (1 - io) * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    dim_proj = out_dim
    rval, __ = theano.scan(_step,
                           sequences=[mask, state_below],
                           outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj * pipe_num),
                                         tensor.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj * pipe_num)],
                           name=_p(prefix, '_layers'),
                           n_steps=nsteps)
    return rval[0]


def multi_blstm_s(tparams, state_below, options, prefix='multi_blstm_s', mask=None, in_dim=None, out_dim=None):
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
        out_dim = options['edim']
    pipe_num = options['pipe_num']
    forget = options['forget']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim * pipe_num:(n + 1) * dim * pipe_num]
        return _x[:, n * dim * pipe_num:(n + 1) * dim * pipe_num]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'f_W')]) + tparams[_p(prefix, 'f_b')]

    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'b_W')]) + tparams[_p(prefix, 'b_b')]

    if options['with_gate'] == True:

        def _step(m_, x_, h_, c_, U):
            preact = tensor.dot(h_, U)
            preact += x_

            io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim)) * forget  # batch_size, pipe_num*edim

            o = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
            c = tensor.tanh(_slice(preact, 2, out_dim))

            c = (io) * c_ + (1 - io) * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c
    else:

        def _step(m_, x_, h_, c_, U):
            preact = tensor.dot(h_, U)
            preact += x_

            io = forget  # batch_size, pipe_num*edim
            o = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
            c = tensor.tanh(_slice(preact, 1, out_dim))

            c = (io) * c_ + (1 - io) * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below_],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num)],
                                non_sequences=[tparams[_p(prefix, 'f_U')]],
                                name=_p(prefix, 'f'),
                                n_steps=nsteps)

    bval, updates = theano.scan(_step,
                                sequences=[mask, state_belowx],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num)],
                                non_sequences=[tparams[_p(prefix, 'b_U')]],
                                name=_p(prefix, 'b'),
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


def multi_blstm_pro(tparams, state_below, options, prefix='multi_blstm_s', mask=None, in_dim=None, out_dim=None):
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
        out_dim = options['edim']
    pipe_num = options['pipe_num']

    ########
    step = options['step']
    bias = options['forget_bias']

    #######
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim * pipe_num:(n + 1) * dim * pipe_num]
        return _x[:, n * dim * pipe_num:(n + 1) * dim * pipe_num]

    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'f_W')]) + tparams[_p(prefix, 'f_b')]

    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'b_W')]) + tparams[_p(prefix, 'b_b')]

    if options['with_gate'] == True:

        def _step(m_, x_, h_, c_, U):
            preact = tensor.dot(h_, U)
            preact += x_

            # io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim)) * tparams['forget']  # batch_size, pipe_num*edim

            io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim)) * step + bias  # batch_size, pipe_num*edim

            o = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
            c = tensor.tanh(_slice(preact, 2, out_dim))

            c = (io) * c_ + (1 - io) * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c
    else:

        def _step(m_, x_, h_, c_, U):
            preact = tensor.dot(h_, U)
            preact += x_

            io = tparams['forget']  # batch_size, pipe_num*edim
            o = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
            c = tensor.tanh(_slice(preact, 1, out_dim))

            c = (io) * c_ + (1 - io) * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            h = o * tensor.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            return h, c

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below_],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num)],
                                non_sequences=[tparams[_p(prefix, 'f_U')]],
                                name=_p(prefix, 'f'),
                                n_steps=nsteps)

    bval, updates = theano.scan(_step,
                                sequences=[mask, state_belowx],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num),
                                              tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           out_dim * pipe_num)],
                                non_sequences=[tparams[_p(prefix, 'b_U')]],
                                name=_p(prefix, 'b'),
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




def multi_lstm_pro(tparams, state_below, options, prefix='multi_lstm_pro', mask=None, in_dim=None, out_dim=None):
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
        out_dim = options['edim']
    pipe_num = options['pipe_num']

    ########
    step = options['step']
    bias = options['forget_bias']

    #######
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim * pipe_num:(n + 1) * dim * pipe_num]
        return _x[:, n * dim * pipe_num:(n + 1) * dim * pipe_num]


    def _step(m_, x_, h_, c_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_

        # io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim)) * tparams['forget']  # batch_size, pipe_num*edim

        io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim)) * step + bias  # batch_size, pipe_num*edim

        o = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        c = tensor.tanh(_slice(preact, 2, out_dim))

        c = (io) * c_ + (1 - io) * c
        c = m_[:, None] * c + (1. - m_)[:, None] * c_

        h = o * tensor.tanh(c)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, c

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])
    dim_proj = out_dim

    rval, __ = theano.scan(_step,
                           sequences=[mask, state_below],
                           outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj * pipe_num),
                                         tensor.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj * pipe_num)],
                           name=_p(prefix, '_layers'),
                           n_steps=nsteps)
    return rval[0]


