__author__ = 'jcxu'

from Util import *
from Modules_gru import *
from Optim import adagrad
from Dataworker import *
import os


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    params = OrderedDict()
    # embedding
    # randn = numpy.random.uniform(low=-.1, high=.1, size=(options['n_words'],
    #                                                      options['dim_proj']))
    # params['Wemb'] = randn.astype(config.floatX)
    # T = ortho_weight(options['dim_proj'], options['dim_hidden'])
    # params['T'] = T

    # params = param_init_gate(options, params, prefix='gate', in_dim=options['dim_hidden'])
    # options, params, prefix='gate', in_dim=None,out_dim=None
    params = param_init_gru(options, params, prefix='gru', nin=options['dim_hidden'], dim=options['dim_hidden'])
    # params = param_init_lstm(options, params, prefix='lstm', in_dim=options['dim_hidden'], out_dim=options['dim_hidden'])

    # params = param_init_lstm(options, params, prefix='lstm_2', in_dim=options['dim_hidden'],
    #                               out_dim=options['dim_hidden'])

    # classifier
    params['U'] = ortho_weight(options['dim_hidden'], options['ydim'])
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def build_model(tparams, options):
    trng = RandomStreams(817)
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('x_mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()]
    emb = emb.reshape([n_timesteps, n_samples, options['dim_proj']])

    # proj = tensor.dot(emb, tparams['T'])
    proj = emb
    proj = gru(tparams, proj, options, prefix='gru', mask=mask,
                     out_dim=options['dim_hidden'])

    proj = dropout_layer(proj, use_noise, trng, options['noise_std'])

    if options['end'] is True:
        proj = proj[-1]
    else:
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6
    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost
