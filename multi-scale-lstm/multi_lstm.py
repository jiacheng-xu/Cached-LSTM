from optim import *
from util import *
from module import *
SEED = 123
numpy.random.seed(SEED)

def init_params(options):
    """
    Global (not LSTM) parameter. For the embeding and the classifier.
    """
    params = OrderedDict()
    # embedding
    # randn = numpy.random.rand(options['n_words'],
    #                           options['dim_proj'])
    # params['Wemb'] = (0.01 * randn).astype(config.floatX)
    params = param_init_multi_lstm(options, params, prefix='multi_lstm', in_dim=options['wdim'], out_dim=options['edim'])

    # classifier
    params['U'] = 0.01 * numpy.random.randn(options['edim'] * options['pipe_num'],
                                            options['ydim']).astype(config.floatX)
    params['b'] = numpy.zeros((options['ydim'],)).astype(config.floatX)

    return params




# ff: Feed Forward (normal neural net), only useful to put after lstm
#     before the classifier.


def build_model(tparams, options):
    trng = RandomStreams(SEED)

    # Used for dropout.
    use_noise = theano.shared(numpy_floatX(0.))

    x = tensor.matrix('x', dtype='int64')
    mask = tensor.matrix('mask', dtype=config.floatX)
    y = tensor.vector('y', dtype='int64')

    n_timesteps = x.shape[0]
    n_samples = x.shape[1]

    emb = tparams['Wemb'][x.flatten()].reshape([n_timesteps,
                                                n_samples,
                                                options['wdim']])

    proj = multi_lstm(tparams, emb, options, mask=mask, prefix='multi_lstm', in_dim=options['wdim'],
                out_dim=options['edim'])


    if options['end'] == True:
        proj = proj[-1]
    else:
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]

    if options['use_dropout']:
        proj = dropout_layer(proj, use_noise, trng,options['noise'])

    pred = tensor.nnet.softmax(tensor.dot(proj, tparams['U']) + tparams['b'])

    f_pred_prob = theano.function([x, mask], pred, name='f_pred_prob')
    f_pred = theano.function([x, mask], pred.argmax(axis=1), name='f_pred')

    off = 1e-8
    if pred.dtype == 'float16':
        off = 1e-6

    cost = -tensor.log(pred[tensor.arange(n_samples), y] + off).mean()

    return use_noise, x, mask, y, f_pred_prob, f_pred, cost

