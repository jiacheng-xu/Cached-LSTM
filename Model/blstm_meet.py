__author__ = 'jcxu'
from Util import *
from Modules import *
from Optim import *
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

    params = param_init_blstm(options, params, prefix='blstm', in_dim=options['dim_proj'], out_dim=options['dim_hidden'])
    params = param_init_weight(options, params, prefix='weight', dim=options['dim_hidden'])
    params = param_init_lstm(options, params, prefix='lstm_2', in_dim=options['dim_hidden'],
                                  out_dim=options['dim_hidden'])

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

    if options['use_dropout']:
        emb = dropout_layer(emb, use_noise, trng, options['noise_std'])

    proj_f, proj_b = blstm(tparams, emb, options, mask=mask, prefix='blstm', in_dim=options['dim_proj'],
                out_dim=options['dim_hidden'])
    proj = weight(tparams, proj_f[:-1],options,proj_b[1:] , prefix='weight')
    proj = lstm(tparams, proj, options, mask=mask[:-1], prefix='lstm_2', in_dim=options['dim_hidden'],
                     out_dim=options['dim_hidden'])

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


def train(dataset='5Label_300_40000_glove.6B',
          n_words=40000,
          decay_c=1e-4,
          optimizer=adagrad,
          clip_c=0.,
          valid_batch_size=64,
          batch_size=25,
          disp_frq=1000,
          valid_freq=100,
          save_freq=1000,
          max_epochs=100,
          lrate=0.05,
          lrate_embed=0.1,
          use_dropout=True,
          noise_std=0.5,
          patience=30,
          saveto='model.npz',
          encoder='lstm',
          dim_proj=300,
          end=True,
          dim_hidden=100
          ):
    # Model options
    model_options = locals().copy()
    print(model_options)
    print 'Loading data'
    path = os.path.join('..', '..', 'Data', 'TC', 'sst_fin', dataset + '.pkl')
    data = pkl.load(open(path, 'rb'))
    train, valid, test, emb = data
    # print(emb.shape)
    ydim = numpy.max(train[1]) + 1

    model_options['ydim'] = ydim

    print 'Building model'
    params = init_params(model_options)

    params['Wemb'] = emb.astype(config.floatX)

    tparams = init_tparams(params)

    (use_noise, x, mask,
     y, f_pred_prob, f_pred, cost) = build_model(tparams, model_options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        for kk, vv in tparams.iteritems():
            if kk is 'Wemb':
                weight_decay += (vv ** 2).sum() / 5.
            else:
                weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost_decay = weight_decay + cost

    f_cost = theano.function([x, mask, y], cost_decay, name='f_cost')

    grads = tensor.grad(cost_decay, wrt=tparams.values())
    """
    if clip_c > 0.:
        g2 = 0.
        for g in grads:
            g2 += (g ** 2).sum()
        new_grads = []
        for g in grads:
            new_grads.append(tensor.switch(g2 > (clip_c ** 2),
                                           g / tensor.sqrt(g2) * clip_c,
                                           g))
        grads = new_grads
    """
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    # lrate_embed = tensor.scalar(name='lrate_embed')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost, cost_decay)

    print 'Optimization'
    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    history_errs = []
    best_p = None
    bad_count = 0

    if valid_freq == -1:
        valid_freq = len(train[0]) / batch_size
    if save_freq == -1:
        save_freq = len(train[0]) / batch_size

    uidx = 0
    estop = False
    start_time = time.time()
    try:
        for eidx in xrange(max_epochs):
            n_samples = 0

            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                if use_dropout is True:
                    use_noise.set_value(1.)
                else:
                    use_noise.set_value(0.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y)
                n_samples += x.shape[1]

                cost, cost_decay = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print 'NaN detected'
                    return 1., 1., 1.

                if numpy.mod(uidx, disp_frq) == 0:
                    print 'Epoch ', eidx, 'Update ', uidx, 'Cost ', cost, 'Cost_decay', cost_decay

                if numpy.mod(uidx, save_freq) == 0:
                    print 'Saving...',

                    # import ipdb; ipdb.set_trace()

                    if best_p != None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                    print 'Done'

                if numpy.mod(uidx, valid_freq) == 0:
                    use_noise.set_value(0.)

                    valid_err = pred_error(f_pred, prepare_data, valid,
                                           kf_valid)
                    history_errs.append(valid_err)

                    if (uidx == 0 or
                                valid_err <= numpy.array(history_errs).min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    if len(history_errs) > patience and valid_err >= numpy.array(history_errs)[:-patience].min():
                        bad_counter += 1
                        if bad_counter > patience:
                            print 'Early Stop!'
                            estop = True
                            break

                    print 'Valid ', valid_err

            print 'Seen %d samples' % n_samples

            if estop:
                break

            if best_p is not None:
                zipp(best_p, tparams)

    except KeyboardInterrupt:
        print "Training interupted"

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    test_err = pred_error(f_pred, prepare_data, test, kf_test)

    print 'Train ', train_err, 'Valid ', valid_err, 'Test ', test_err

    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))
    return train_err, valid_err, test_err


# layers = {'lstm': (param_init_lstm, lstm), 'lstm_att': (param_init_lstm_att, lstm_att),
#           'blstm': (param_init_blstm, blstm), 'plstm': (param_init_plstm, plstm)}

if __name__ == '__main__':
    train_err, valid_err, test_err = train(
        dataset='5Label_300_40000_glove.6B',
        dim_proj=300,
        dim_hidden=64,

        max_epochs=80,
        use_dropout=True,
        noise_std=0.2,

        patience=15,
        optimizer=adagrad,
        decay_c=1e-4,
        disp_frq=2000,
        valid_freq=2000,
        batch_size=25
    )
