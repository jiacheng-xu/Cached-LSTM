__author__ = 'jcxu'
# from FullGRU_3 import *
import sys
from collections import OrderedDict
import theano
import theano.tensor as tensor
import numpy
from Util import *
# from template import *
from Optim import adagrad
import os

decay = float(sys.argv[1])
drop = float(sys.argv[2])
dataname = sys.argv[3]
name = sys.argv[4]

# name = 'gru_5_nodes'
if name == '3':
    from TGRU_3 import *
elif name =='5':
    from TGRU_5 import *
else:
    print 'Import error, exit'
    sys.exit()

data = dataname + '_50_100000_glove.twitter.27B'
# dataname='Yelp2013'

decay_list = [1e-5, 1e-4]
drop_list = [0, 0.3, 0.5]
hidden = 50


def train(
        dataname='5r',
        dataset='5Label_300_40000_glove.6B',
        n_words=40000,
        decay_c=1e-4,
        optimizer=adagrad,
        clip_c=4.,
        valid_batch_size=64,
        batch_size=32,
        disp_frq=1000,
        valid_freq=100,
        save_freq=1000,
        max_epochs=100,
        # lrate=0.05,
        lrate=0.01,
        lrate_embed=0.1,
        use_dropout=True,
        noise_std=0.5,
        patience=15,
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

    path = os.path.join('..','..', '..', 'Data', 'TC', dataname, dataset + '.pkl')
    # path = os.path.join('..', '..', 'Data', 'TC', dataname, dataset + '.pkl')
    data = pkl.load(open(path, 'rb'))
    train, valid, test, emb = data
    print(emb.shape)
    ydim = numpy.max(train[1]) - numpy.min(train[1]) + 1

    if numpy.min(train[1]) is not 0:
        bias = numpy.min(train[1])
        print 'Min of class is ', bias

        def min_y_to_zero(set):
            X, Y = set[0], set[1]
            new_Y = []
            for y in Y:
                new_Y.append(y - bias)
            return [X, new_Y]

        train = min_y_to_zero(train)
        valid = min_y_to_zero(valid)
        test = min_y_to_zero(test)

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
            # weight_decay+=(theano.ifelse(kk is 'Wemb'), ((vv ** 2).sum() / 5.), ((vv ** 2).sum()))
            # if kk is 'Wemb':
            #     weight_decay += (vv ** 2).sum() / 5.
            # else:
            #     weight_decay += (vv ** 2).sum()
            weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost_decay = weight_decay + cost

    f_cost = theano.function([x, mask, y], cost_decay, name='f_cost')

    grads = tensor.grad(cost_decay, wrt=tparams.values())

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

    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    # lrate_embed = tensor.scalar(name='lrate_embed')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost, cost_decay)

    print 'Optimization'
    # kf_train4valid = get_minibatches_idx(len(train4valid[0]), valid_batch_size)
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
    test_err = pred_error(f_pred, prepare_data, test, kf_test)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    valid_err = pred_error(f_pred, prepare_data, valid, kf_valid)
    print 'Valid ', valid_err, 'Test ', test_err

    train_err = pred_error(f_pred, prepare_data, train, kf_train_sorted)

    # print 'Train ', train_err,'Train4Valid ',train4valid_err, 'Valid ', valid_err, 'Test ', test_err
    print 'Train ', train_err
    print 'Dataset', dataname, 'Test Acc', (1. - test_err)
    print(model_options)
    if saveto:
        numpy.savez(saveto, train_err=train_err,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)
    print 'The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1)))
    print >> sys.stderr, ('Training took %.1fs' %
                          (end_time - start_time))

    return train_err, valid_err, test_err


train_err, valid_err, test_err = train(
    dataname=dataname,
    dataset=data,
    dim_proj=50,
    dim_hidden=hidden,
    max_epochs=40,
    use_dropout=True,
    noise_std=drop,
    patience=15,
    optimizer=adagrad,
    decay_c=decay,
    disp_frq=1000,
    valid_freq=-1,
    batch_size=16,
    lrate=0.05,
    lrate_embed=0.1,
    saveto=name,
    end=True
)
if os.path.isfile((name + '.md')) is False:
    f = open((name + '.md'), 'w')
    f.close()

f = open((name + '.md'), 'a')
f.write('|' + str(decay) + '|' + data + '|' + str(drop) + '|' + str(
    1. - train_err)
        # +'|' + str(1. - train4valid)
        + '|' + str(1. - valid_err) + '|' + str(1. - test_err) + '|\n')
f.close()
