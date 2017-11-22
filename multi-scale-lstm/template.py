'''
Build a tweet sentiment analyzer
'''

from __future__ import print_function
import six.moves.cPickle as pickle

from collections import OrderedDict
import sys
import time
import getopt
import sys
from optim import *
from module import *
from util import *
import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# Set the random number generators' seeds for consistency
SEED = 76767
numpy.random.seed(SEED)


def handle_argv(argv):
    opts, args = getopt.getopt(argv[1:], 'm:d:l:n:p:s:',
                               ['dataset=', 'name=', 'load=', 'edim=', 'wdim=', 'decay=',
                                'lr=', 'optim=', 'pp_num=', 'max_len=', 'batch=',
                                'pp_rate=', 'pp_decay=', 'pp_list=', 'gate=', 'lr_fgt=',
                                'epo=', 'end=', 'test=','drop='])
    print(opts)
    dataset = 'qc'
    model_name = 'multi_lstm_pro'
    max_len = None
    batch_size = 32
    optim = 'adagrad'
    reload = None
    decay = 1e-4
    edim = 120
    wdim = 50
    lrate = 0.01
    pipe_num = 1
    pipe_list = [1.]
    pipe_rate = 1.0
    pipe_decay = 0.5
    gate = True
    lr_forget = 0.01
    epo = 20
    end = False
    test = True
    drop=0.5
    # lrtxt = 'lrate'
    for o, a in opts:
        if o == '--dataset':
            dataset = str(a)
        elif o == '--load':
            reload = str(a)
        elif o == '--lr':
            lrate = float(a)
        elif o == '--name':
            model_name = str(a)
        elif o == '--max_len':
            max_len = int(a)
        elif o == '--epo':
            epo = int(a)
        elif o == '--batch':
            batch_size = int(a)
        elif o == '--edim':
            edim = int(a)
        elif o == '--wdim':
            wdim = int(a)
        elif o == '--pp_rate':
            pipe_rate = float(a)
        elif o == '--pp_decay':
            pipe_decay = float(a)
        elif o == '--pp_num':
            pipe_num = int(a)
        elif o == '--pp_list':
            pipe_list = list(a)
        elif o == '--decay':
            decay = float(a)
        elif o == '--optim':
            optim = a
        elif o == '--gate':
            gate = bool(a)
        elif o == '--lr_fgt':
            lr_forget = float(a)
        elif o == '--end':
            end = bool(a)
        elif o == '--test':
            test = bool(a)
        elif o=='drop':
            drop=float(a)
    train(dataset=dataset, reload_model=reload, lrate=lrate, model_name=model_name,
          pipe_rate=pipe_rate, max_len=max_len, batch_size=batch_size, edim=edim, wdim=wdim,
          pipe_decay=pipe_decay, pipe_list=pipe_list, pipe_num=pipe_num, decay_c=decay
          , optim=optim, with_gate=gate, lr_forget=lr_forget, max_epochs=epo, end=end,
          test=test,noise=drop)


def _p(pp, name):
    return '%s_%s' % (pp, name)


# def test():
#     pass


def train(
        lr_forget=0.,
        with_gate=True,
        test=False,
        model_name='lstm',
        # Embedding dim
        edim=32,  # word embeding dimension and LSTM number of hidden units.
        wdim=50,
        optim='sgd',
        patience=20,  # Number of epoch to wait before early stop if no progress
        max_epochs=20,  # The maximum number of epoch to run

        disp_freq=800,  # Display to stdout the training progress every N updates
        valid_freq=-1,  # Compute the validation error after this number of update.
        save_freq=4000,  # Save the parameters after every saveFreq updates

        end=False,

        decay_c=1e-4,  # Weight decay for the classifier applied to the U weights.

        lrate=0.001,  # Learning rate for sgd (not used for adadelta and rmsprop)
        lrtxt='lrate',

        optimizer=adagrad,  # adagrad, sgd, adadelta and rmsprop available, sgd very hard to use,
        #  not recommanded (probably need momentum and decaying learning rate).

        pipe_num=1,
        pipe_rate=0.9,
        pipe_decay=0.5,
        pipe_list=None,
        max_len=None,  # Sequence longer then this get ignored

        batch_size=128,  # The batch size during training.
        valid_batch_size=512,  # The batch size used for validation/test set.

        dataset='imdb2w',  # imdb25000.pkl

        # Parameter for extra option
        noise=.5,
        use_dropout=True,

        reload_model=None,  # Path to a saved model we want to start from.
        valid_size=8192,
):
    # Model options

    if optim == 'adagrad':
        optimizer = adagrad
    elif optim == 'sgd':
        optimizer = sgd
    elif optim == 'adadelta':
        optimizer = adadelta
    elif optim == 'adam':
        optimizer = adam
    elif optim == 'rmsprop':
        optimizer = rmsprop
    else:
        raise NotImplementedError

    edim = int(edim / pipe_num)
    options = locals().copy()
    print("Model options:", options)

    print('-----Loading data-----')
    import os
    path = os.path.join('..', 'data', dataset + '.pkl')
    train, valid, test, emb = pkl.load(open(path, 'rb'))

    print('Vocb shape:', emb.shape)
    options['n_words'] = emb.shape[0]
    assert wdim == emb.shape[1]

    saveto = ('%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s_%s.npz' % (dataset, model_name, str(lrate),
                                                        optimizer.__name__, str(noise), str(edim), str(wdim),
                                                        str(pipe_num), str(with_gate), str(decay_c), str(batch_size),str(noise)
                                                        ))
    print('------------SAVE TO:%s------------' % (saveto))
    options['saveto'] = saveto
    # num of classification
    ydim = numpy.max(train[1]) - numpy.min(train[1]) + 1
    print(ydim)
    options['ydim'] = ydim

    if numpy.min(train[1]) != 0:
        bias = numpy.min(train[1])
        print('Min of class is ', bias)

        def min_y_to_zero(set):
            X, Y = set[0], set[1]
            new_Y = []
            for y in Y:
                new_Y.append(y - bias)
            return [X, new_Y]

        train = min_y_to_zero(train)
        valid = min_y_to_zero(valid)
        test = min_y_to_zero(test)

    # valid = test

    #######
    if valid_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = numpy.arange(len(valid[0]))
        numpy.random.shuffle(idx)
        idx = idx[:valid_size]
        valid = ([valid[0][n] for n in idx], [valid[1][n] for n in idx])
    #######


    print('Building model')
    if model_name == 'lstm':
        from lstm import init_params, build_model
    elif model_name == 'lstm_s':
        from lstm_s import init_params, build_model
    elif model_name == 'multi_lstm':
        from multi_lstm import init_params, build_model
    elif model_name == 'multi_lstm_s':
        from multi_lstm_s import init_params, build_model
    elif model_name == 'cbow':
        from cbow import init_params, build_model
    elif model_name == 'rnn':
        from rnn import init_params, build_model
    elif model_name == 'blstm':
        from blstm import init_params, build_model
    elif model_name == 'blstm_s':
        from blstm_s import init_params, build_model
    elif model_name == 'multi_blstm_s':
        from multi_blstm_s import init_params, build_model
    elif model_name == 'multi_blstm_pro':
        from multi_blstm_pro import init_params, build_model
    elif model_name == 'multi_lstm_pro':
        from multi_lstm_pro import init_params,build_model
    else:
        raise NotImplementedError

    params = init_params(options)
    params['Wemb'] = emb.astype(config.floatX)

    if reload_model:
        load_params(reload_model, params)

    # This create Theano Shared Variable from the parameters.
    # Dict name (string) -> Theano Tensor Shared Variable
    # params and tparams have different copy of the weights.
    tparams = init_tparams(params)

    # use_noise is for dropout
    (use_noise, x, mask, y, f_pred_prob, f_pred, cost, proj) = build_model(tparams, options)

    if decay_c > 0.:
        decay_c = theano.shared(numpy_floatX(decay_c), name='decay_c')
        weight_decay = 0.
        # TODO
        weight_decay += (tparams['U'] ** 2).sum()
        # for kk, vv in tparams.iteritems():
        #     weight_decay += (vv ** 2).sum()
        weight_decay *= decay_c
        cost += weight_decay

    f_monitor = theano.function([x, mask], proj, name='monitor', on_unused_input='ignore')

    f_cost = theano.function([x, mask, y], cost, name='f_cost')

    grads = tensor.grad(cost, wrt=list(tparams.values()))
    # TODO Gradient Clip
    f_grad = theano.function([x, mask, y], grads, name='f_grad')

    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = optimizer(lr, tparams, grads,
                                        x, mask, y, cost)

    print('Optimization')

    kf_valid = get_minibatches_idx(len(valid[0]), valid_batch_size)
    kf_test = get_minibatches_idx(len(test[0]), valid_batch_size)

    print("%d train examples" % len(train[0]))
    print("%d valid examples" % len(valid[0]))
    print("%d test examples" % len(test[0]))

    history_errs = []
    best_p = None
    bad_count = 0

    if valid_freq == -1:
        valid_freq = len(train[0]) // batch_size
    if save_freq == -1:
        saveFreq = len(train[0]) // batch_size

    uidx = -1  # the number of update done
    estop = False  # early stop
    start_time = time.time()
    try:
        for eidx in range(max_epochs):
            n_samples = 0

            # options['max_len'] = 60 + 20 *eidx

            # Get new shuffled index for the training set.
            kf = get_minibatches_idx(len(train[0]), batch_size, shuffle=True)

            for _, train_index in kf:
                uidx += 1
                use_noise.set_value(1.)

                # Select the random examples for this minibatch
                y = [train[1][t] for t in train_index]
                x = [train[0][t] for t in train_index]

                # Get the data in numpy.ndarray format
                # This swap the axis!
                # Return something of shape (minibatch maxlen, n samples)
                x, mask, y = prepare_data(x, y, maxlen=options['max_len'])
                n_samples += x.shape[1]

                cost = f_grad_shared(x, mask, y)
                f_update(lrate)

                if numpy.isnan(cost) or numpy.isinf(cost):
                    print('bad cost detected: ', cost)
                    return 1., 1., 1.

                if numpy.mod(uidx, disp_freq) == 0:
                    proj = f_monitor(x, mask)
                    print('Epoch ', eidx, 'Update ', uidx, 'Cost ', cost,
                          ' Time0:', numpy.abs(proj[0]).mean(), ' Time-1:', numpy.abs(proj[-1]).mean()
                          )

                if saveto and numpy.mod(uidx, save_freq) == 0:
                    print('Saving...')

                    if best_p is not None:
                        params = best_p
                    else:
                        params = unzip(tparams)
                    numpy.savez(saveto, history_errs=history_errs, **params)
                    pickle.dump(options, open('%s.pkl' % saveto, 'wb'), -1)
                    print('Done')

                if numpy.mod(uidx, valid_freq) == 0:
                    print(dataset)
                    use_noise.set_value(0.)
                    # train_err = pred_error(f_pred, prepare_data, train, kf, options)
                    valid_err, mse_err = pred_error(f_pred, prepare_data, valid,
                                                    kf_valid, options)
                    # visual(f_monitor,prepare_data,valid,kf_valid,options)
                    # mse_err = pred_mse(f_pred_prob,prepare_data,valid,kf_valid,options)
                    # test_err = pred_error(f_pred, prepare_data, test, kf_test, options)

                    history_errs.append([valid_err, mse_err])

                    if (best_p is None or
                                valid_err <= numpy.array(history_errs)[:,
                                             0].min()):
                        best_p = unzip(tparams)
                        bad_counter = 0

                    print(('Data:', dataset, 'Valid ', valid_err, 'MSE:', mse_err))

                    if (len(history_errs) > patience and
                                valid_err >= numpy.array(history_errs)[:-patience,
                                             0].min()):
                        bad_counter += 1
                        if bad_counter > patience:
                            print('Early Stop!')
                            estop = True
                            break

            print('Seen %d samples' % n_samples)

            if estop:
                break

    except KeyboardInterrupt:
        print("Training interupted")

    end_time = time.time()
    if best_p is not None:
        zipp(best_p, tparams)
    else:
        best_p = unzip(tparams)

    if test:
        part_test_len = int(len(test[0]) / 10)
        print(part_test_len)
        use_noise.set_value(0.)
        # Split dataset
        dict_len = {}
        total_err, total_mse = 0., 0.
        for idx in xrange(len(test[0])):
            dict_len[idx] = len(test[0][idx])
        sorted_list = sorted(dict_len.iteritems(), key=lambda e: e[1])
        for i in xrange(10):
            tmp_list = sorted_list[i * part_test_len:(i + 1) * part_test_len]
            tmp_len = len(tmp_list)
            test_set_x = []
            test_set_y = []
            for j in tmp_list:
                index = j[0]
                test_set_x.append(test[0][index])
                test_set_y.append(test[1][index])
            kf = get_minibatches_idx(tmp_len, valid_batch_size)
            test_set = [test_set_x, test_set_y]
            valid_err, mse_err = pred_error(f_pred, prepare_data, test_set,
                                            kf, options)
            total_err += valid_err
            total_mse += mse_err
            print('%s~%s: %s\t%s' % (i * 10, (i + 1) * 10, valid_err, mse_err))
        total_err = total_err / 10.
        total_mse /= 10.
        print('Total:%s %s' % (total_err, total_mse))



    use_noise.set_value(0.)
    kf_train_sorted = get_minibatches_idx(len(train[0]), batch_size)
    train_err, train_mse_err = pred_error(f_pred, prepare_data, train, kf_train_sorted, options)
    valid_err, valid_mse_err = pred_error(f_pred, prepare_data, valid, kf_valid, options)
    test_err, test_mse_err = pred_error(f_pred, prepare_data, test, kf_test, options)
    history_errs.append([test_err, test_mse_err])
    # valid_mse_err = pred_mse(f_pred_prob,prepare_data,valid,kf_valid,options)
    # test_mse_err = pred_mse(f_pred_prob,prepare_data,test,kf_test,options)
    print(options)

    print('Train ', train_err, 'Valid ', valid_err, 'Test ', test_err)
    print('MSE: Train ', train_mse_err, 'Valid ', valid_mse_err, 'Test ', test_mse_err)
    if saveto:
        numpy.savez(saveto,
                    valid_err=valid_err, test_err=test_err,
                    history_errs=history_errs, **best_p)

    print('The code run for %d epochs, with %f sec/epochs' % (
        (eidx + 1), (end_time - start_time) / (1. * (eidx + 1))))
    print(('Training took %.1fs' %
           (end_time - start_time)), file=sys.stderr)
    return valid_err, test_err


if __name__ == '__main__':
    # See function train for all possible parameter and there definition.
    # train(model_name='lstm_s',
    #       max_epochs=100,pipe_rate=0.9
    #       )
    handle_argv(argv=sys.argv)
