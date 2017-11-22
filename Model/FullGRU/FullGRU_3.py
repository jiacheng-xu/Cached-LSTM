__author__ = 'jcxu'

from Util import *
from Modules_gru import *
from Optim import adagrad
# from Dataworker import *
import os


# make prefix-appended name
def _p(pp, name):
    return '%s_%s' % (pp, name)


def init_params(options):
    params = OrderedDict()
    # T = ortho_weight(options['dim_proj'], options['dim_hidden'])
    # params['T'] = T

    params = param_init_gru_conv(options, params, prefix='gru_conv_11', nin=options['dim_hidden'], dim=options['dim_hidden'])
    params = param_init_gru_conv(options, params, prefix='gru_conv_12', nin=options['dim_hidden'], dim=options['dim_hidden'])

    params = param_init_gru_conv(options, params, prefix='gru_conv_21', nin=options['dim_hidden'], dim=options['dim_hidden'])
    # params = param_init_gru_conv(options, params, prefix='gru_conv_22', nin=options['dim_hidden'], dim=options['dim_hidden'])

    # classifier
    params['U'] = ortho_weight(3 * options['dim_hidden'], options['ydim'])
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params


def build_model(tparams, options):
    trng = RandomStreams(1206)
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

    proj_11 = gru_conv_naive(tparams, proj[:-1], options, prefix='gru_conv_11', mask=mask[:-2], out_dim=options['dim_hidden'])
    proj_12 = gru_conv_naive(tparams, proj[1:], options, prefix='gru_conv_12', mask=mask[1:-1], out_dim=options['dim_hidden'])
    # proj_13 = gru_conv(tparams, proj[2:-1], options, prefix='gru_conv_13', mask=mask[2:-1], out_dim=options['dim_hidden'])
    # proj_14 = gru_conv(tparams, proj, options, prefix='gru_conv_14', mask=mask[3:-1], out_dim=options['dim_hidden'])

    proj_11 = dropout_layer(proj_11, use_noise, trng, options['noise_std'])
    proj_12 = dropout_layer(proj_12, use_noise, trng, options['noise_std'])

    proj_21 = gru_conv(tparams, proj_11,proj_12, options, prefix='gru_conv_21', mask=mask[:-2], out_dim=options['dim_hidden'])

    proj = dropout_layer(proj_21, use_noise, trng, options['noise_std'])

    if options['end'] is True:
        proj = concatenate((proj_11[-1], proj_12[-1], proj[-1]), axis=1)
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
