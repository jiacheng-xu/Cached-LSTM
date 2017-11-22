from optim import *
from util import *


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


def param_init_lstm(options, params, prefix='lstm', in_dim=None, out_dim=None):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['edim']
    W = numpy.concatenate([glorot_uniform(in_dim, out_dim),
                           glorot_uniform(in_dim, out_dim),
                           glorot_uniform(in_dim, out_dim),
                           glorot_uniform(in_dim, out_dim, 4.)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((4 * out_dim,))
    b[out_dim:2 * out_dim] = 1.
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm(tparams, state_below, options, prefix='lstm', mask=None, in_dim=None, out_dim=None):
    if out_dim is None:
        out_dim = options['edim']
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

        c = options['pipe_rate'] * f * c_ + i * c
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
                                                      dim_proj),
                                         tensor.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj)],
                           name=_p(prefix, '_layers'),
                           n_steps=nsteps)
    return rval[0]


def param_init_lstm_s(options, params, prefix='lstm', in_dim=None, out_dim=None):
    """
    Init the LSTM parameter:

    :see: init_params
    """
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['edim']
    W = numpy.concatenate([glorot_uniform(in_dim, out_dim),
                           glorot_uniform(in_dim, out_dim),
                           glorot_uniform(in_dim, out_dim, 4.)], axis=1)
    params[_p(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim),
                           ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'U')] = U
    b = numpy.zeros((3 * out_dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)

    return params


def lstm_s(tparams, state_below, options, prefix='lstm', mask=None, in_dim=None, out_dim=None):
    if out_dim is None:
        out_dim = options['edim']
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

        io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        c = tensor.tanh(_slice(preact, 2, out_dim))

        c = (1 - io) * c_ + io * c
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
                                                      dim_proj),
                                         tensor.alloc(numpy_floatX(0.),
                                                      n_samples,
                                                      dim_proj)],
                           name=_p(prefix, '_layers'),
                           n_steps=nsteps)
    return rval[0]


def param_init_multi_lstm(options, params, prefix='multi_lstm', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['edim']

    list_W, list_U = [], []
    # TODO 4 for lstm
    for i in xrange(options['pipe_num'] * 4):
        list_W.append(glorot_uniform(in_dim, out_dim))
        list_U.append(ortho_weight(options['pipe_num'] * out_dim, out_dim))
    params[_p(prefix, 'W')] = numpy.concatenate(list_W, axis=1)
    params[_p(prefix, 'U')] = numpy.concatenate(list_U, axis=1)
    params[_p(prefix, 'b')] = numpy.zeros((4 * options['pipe_num'] * out_dim,)).astype(config.floatX)

    return params


def multi_lstm(tparams, state_below, options, prefix='multi_lstm', mask=None, in_dim=None, out_dim=None):
    if out_dim is None:
        out_dim = options['edim']
    pipe_num = options['pipe_num']
    forget_mat = options['forget_matrix']
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

        i = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))  # batch_size, pipe_num*edim
        f = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 2, out_dim))
        c = tensor.tanh(_slice(preact, 3, out_dim))

        f = f * forget_mat

        c = f * c_ + i * c
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


def param_init_multi_lstm_s(options, params, prefix='multi_lstm', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['edim']

    list_W, list_U = [], []
    # TODO 4 for lstm
    for i in xrange(options['pipe_num'] * 3):
        list_W.append(glorot_uniform(in_dim, out_dim))
        list_U.append(ortho_weight(options['pipe_num'] * out_dim, out_dim))
    params[_p(prefix, 'W')] = numpy.concatenate(list_W, axis=1)
    params[_p(prefix, 'U')] = numpy.concatenate(list_U, axis=1)
    params[_p(prefix, 'b')] = numpy.zeros((3 * options['pipe_num'] * out_dim,)).astype(config.floatX)

    return params


def multi_lstm_s(tparams, state_below, options, prefix='multi_lstm', mask=None, in_dim=None, out_dim=None):
    if out_dim is None:
        out_dim = options['edim']
    pipe_num = options['pipe_num']
    forget_mat = options['forget_matrix']
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

        io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim)) * forget_mat  # batch_size, pipe_num*edim
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
                   tensor.dot(state_below[1:], tparams[_p(prefix, 'V')]) + \
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
                                n_steps=nsteps - 1)
    # profile=profile,
    # strict=True)
    # rval = [rval]
    return rval  # GRU layer


def param_init_rnn(options, params, prefix='rnn', in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['edim']

    params[_p(prefix, 'W')] = glorot_uniform(in_dim, out_dim)
    params[_p(prefix, 'U')] = ortho_weight(out_dim, out_dim)
    b = numpy.zeros((out_dim,))
    params[_p(prefix, 'b')] = b.astype(config.floatX)
    return params


def rnn(tparams, state_below, options, prefix='rnn', mask=None, in_dim=None, out_dim=None):
    if in_dim is None:
        in_dim = options['wdim']
    if out_dim is None:
        out_dim = options['edim']

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    assert mask is not None

    def _step(m_, x_, h_):
        preact = tensor.dot(h_, tparams[_p(prefix, 'U')])
        preact += x_
        h = tensor.nnet.sigmoid(preact)
        h = m_[:, None] * h + (1. - m_)[:, None] * h_
        return h

    state_below = (tensor.dot(state_below, tparams[_p(prefix, 'W')]) +
                   tparams[_p(prefix, 'b')])

    dim_proj = out_dim

    rval, updates = theano.scan(_step,
                                sequences=[mask, state_below],
                                outputs_info=[tensor.alloc(numpy_floatX(0.),
                                                           n_samples,
                                                           dim_proj)],
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps)
    return rval[0]



def param_init_blstm(options, params, prefix='blstm', in_dim=None, out_dim=None):
    """
    Use weights between forward and backward.
    """
    if in_dim is None:
        in_dim = options['dim_proj']
    if out_dim is None:
        out_dim = options['dim_proj']

    Wf = numpy.concatenate([glorot_uniform(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim,4.)], axis=1)
    params[_p(prefix, 'Wf')] = Wf
    Uf = numpy.concatenate([ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            glorot_uniform(out_dim, out_dim),
                            glorot_uniform(out_dim, out_dim,4.)], axis=1)
    params[_p(prefix, 'Uf')] = Uf
    bf = numpy.zeros((4 * out_dim,))
    bf[out_dim:2 * out_dim] = 1.
    params[_p(prefix, 'bf')] = bf.astype(config.floatX)

    Wb = numpy.concatenate([glorot_uniform(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim,4.)], axis=1)
    params[_p(prefix, 'Wb')] = Wb
    Ub = numpy.concatenate([ortho_weight(out_dim, out_dim),
                            glorot_uniform(out_dim, out_dim),
                            glorot_uniform(out_dim, out_dim),
                            glorot_uniform(out_dim, out_dim,4.)], axis=1)
    params[_p(prefix, 'Ub')] = Ub
    bb = numpy.zeros((4 * out_dim,))
    bb[out_dim:2 * out_dim] = 1.
    params[_p(prefix, 'bb')] = bb.astype(config.floatX)

    Vf = numpy.concatenate([glorot_uniform(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vf')] = Vf
    Vb = numpy.concatenate([glorot_uniform(out_dim, out_dim)], axis=1)
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
        out_dim = options['edim']

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



def param_init_blstm_s(options, params, prefix='blstm_s', in_dim=None, out_dim=None):
    """
    Use weights between forward and backward.
    """
    if in_dim is None:
        in_dim = options['dim_proj']
    if out_dim is None:
        out_dim = options['dim_proj']

    Wf = numpy.concatenate([
                            glorot_uniform(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim,4.)], axis=1)
    params[_p(prefix, 'Wf')] = Wf
    Uf = numpy.concatenate([
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim),
                            ortho_weight(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Uf')] = Uf
    bf = numpy.zeros((3 * out_dim,))
    params[_p(prefix, 'bf')] = bf.astype(config.floatX)

    Wb = numpy.concatenate([
                            ortho_weight(in_dim, out_dim),
                            ortho_weight(in_dim, out_dim),
                            glorot_uniform(in_dim, out_dim,4.)], axis=1)
    params[_p(prefix, 'Wb')] = Wb
    Ub = numpy.concatenate([glorot_uniform(out_dim, out_dim),
                            glorot_uniform(out_dim, out_dim),
                            glorot_uniform(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Ub')] = Ub
    bb = numpy.zeros((3 * out_dim,))
    params[_p(prefix, 'bb')] = bb.astype(config.floatX)

    Vf = numpy.concatenate([glorot_uniform(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vf')] = Vf
    Vb = numpy.concatenate([glorot_uniform(out_dim, out_dim)], axis=1)
    params[_p(prefix, 'Vb')] = Vb
    bo = numpy.zeros((out_dim,)).astype(config.floatX)
    params[_p(prefix, 'bo')] = bo
    return params


def blstm_s(tparams, state_below, options, prefix='blstm', mask=None, in_dim=None, out_dim=None):
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

        io = tensor.nnet.sigmoid(_slice(preact, 0, out_dim))
        o = tensor.nnet.sigmoid(_slice(preact, 1, out_dim))
        c = tensor.tanh(_slice(preact, 2, out_dim))

        c = (1 - io) * c_ + io * c
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

