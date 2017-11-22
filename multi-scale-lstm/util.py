from optim import *
from util import *

from collections import OrderedDict
import cPickle as pkl
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.items():
        if kk not in pp:
            print('%s is not in the archive' % kk)
        else:
            if params[kk].shape == pp[kk].shape:
                params[kk] = pp[kk]
            else:
                print('Shape not equal %s: init:%s load: %s'%(kk,params[kk].shape, pp[kk].shape))
    return params


def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.items():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


def prepare_data(seqs, labels, maxlen=None):
    """Create the matrices from the datasets.

    This pad each sequence to the same lenght: the lenght of the
    longuest sequence or maxlen.

    if maxlen is set, we will cut all sequence to this maximum
    lenght.

    This swap the axis!
    """
    # x: a list of sentences
    lengths = [len(s) for s in seqs]

    if maxlen is not None:
        new_seqs = []
        new_labels = []
        new_lengths = []
        for l, s, y in zip(lengths, seqs, labels):
            if l < maxlen:
                new_seqs.append(s)
                new_labels.append(y)
                new_lengths.append(l)
            else:
                new_seqs.append(s[:maxlen])
                new_labels.append(y)
                new_lengths.append(maxlen)
        lengths = new_lengths
        labels = new_labels
        seqs = new_seqs

        if len(lengths) < 1:
            return None, None, None

    n_samples = len(seqs)
    maxlen = numpy.max(lengths)

    x = numpy.zeros((maxlen, n_samples)).astype('int64')
    x_mask = numpy.zeros((maxlen, n_samples)).astype(theano.config.floatX)
    for idx, s in enumerate(seqs):
        x[:lengths[idx], idx] = s
        x_mask[:lengths[idx], idx] = 1.

    return x, x_mask, labels


def get_minibatches_idx(n, minibatch_size, shuffle=False):
    """
    Used to shuffle the dataset at each iteration.
    """

    idx_list = numpy.arange(n, dtype="int32")

    if shuffle:
        numpy.random.shuffle(idx_list)

    minibatches = []
    minibatch_start = 0
    for i in range(n // minibatch_size):
        minibatches.append(idx_list[minibatch_start:
        minibatch_start + minibatch_size])
        minibatch_start += minibatch_size

    if (minibatch_start != n):
        # Make a minibatch out of what is left
        minibatches.append(idx_list[minibatch_start:])

    return zip(range(len(minibatches)), minibatches)


def zipp(params, tparams):
    """
    When we reload the model. Needed for the GPU stuff.
    """
    for kk, vv in params.items():
        tparams[kk].set_value(vv)


def unzip(zipped):
    """
    When we pickle the model. Needed for the GPU stuff.
    """
    new_params = OrderedDict()
    for kk, vv in zipped.items():
        new_params[kk] = vv.get_value()
    return new_params


def dropout_layer(state_before, use_noise, trng, noise=0.5):
    proj = tensor.switch(use_noise,
                         (state_before *
                          trng.binomial(state_before.shape,
                                        p=(1 - noise), n=1,
                                        dtype=state_before.dtype)),
                         state_before * (1 - noise))
    return proj


def glorot_uniform(in_dim, out_dim=None, const=1.0):
    if out_dim == None:
        out_dim = in_dim
    s = numpy.sqrt(6. / (in_dim + out_dim)) * const
    w = numpy.random.uniform(low=-s, high=s, size=(in_dim, out_dim)).astype(config.floatX)
    return w


def ortho_weight(in_dim, out_dim=None):
    if out_dim is None:
        out_dim = in_dim

    shape = (in_dim, out_dim)
    flat_shape = (shape[0], numpy.prod(shape[1:]))
    a = numpy.random.normal(0.0, 1.0, flat_shape)
    u, _, v = numpy.linalg.svd(a, full_matrices=False)
    # pick the one with the correct shape
    q = u if u.shape == flat_shape else v
    q = q.reshape(shape)
    return (1.1 * q[:shape[0], :shape[1]]).astype(config.floatX)


def pred_probs(f_pred_prob, prepare_data, data, iterator, options, verbose=False):
    """ If you want to use a trained model, this is useful to compute
    the probabilities of new examples.
    """
    n_samples = len(data[0])
    probs = numpy.zeros((n_samples, 2)).astype(config.floatX)

    n_done = 0

    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=options['max_len'])
        pred_probs = f_pred_prob(x, mask)
        probs[valid_index, :] = pred_probs

        n_done += len(valid_index)
        if verbose:
            print('%d/%d samples classified' % (n_done, n_samples))

    return probs


def pred_error(f_pred, prepare_data, data, iterator, options, verbose=False):
    """
    Just compute the error
    f_pred: Theano fct computing the prediction
    prepare_data: usual prepare_data for that dataset.
    """
    valid_err = 0
    mse_error = 0.
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=options['max_len'])
        preds = f_pred(x, mask)
        targets = numpy.array(data[1])[valid_index]
        valid_err += (preds == targets).sum()
        mse_error += ((preds - targets) ** 2).sum()

    valid_err = 1. - numpy_floatX(valid_err) / len(data[0])
    mse_error = numpy_floatX(mse_error) / len(data[0])
    # print(mse_error)
    return valid_err, mse_error


def visual(f_monitor, prepare_data, data, iterator, options):
    n_samples = len(data[0])

    rt_embedding = numpy.zeros((n_samples, options['edim']),dtype='float')
    rt_targets = numpy.zeros((n_samples,),dtype='int')
    cursor = 0
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=options['max_len'])
        print(x)
        print(mask)
        batch_size = len(y)
        proj = f_monitor(x, mask)
        print(proj.shape)
        # if options['end'] == True:
        #     proj = proj[-1]
        # else:
        proj = (proj * mask[:, :, None]).sum(axis=0)
        proj = proj / mask.sum(axis=0)[:, None]

        # print(proj.shape)
        assert proj.shape[0]==batch_size
        assert proj.shape[1]==options['edim']

        # print('Before')
        # print(rt_embedding[cursor:cursor+batch_size,:])
        # print(proj)
        rt_embedding[cursor:cursor+batch_size,:] = proj
        print('After')
        print(rt_embedding[cursor:cursor+batch_size,:])
        targets = numpy.array(data[1])[valid_index]

        rt_targets[cursor:cursor+batch_size] = targets

        cursor+=batch_size

    print('Visual Done!')
    print rt_embedding.shape
    print rt_targets.shape
    numpy.savez('visual'+options['saveto'], embedding=rt_embedding,target=rt_targets)

def pred_mse(f_pred_prob, prepare_data, data, iterator, options, verbose=False):
    mse_error = 0.
    for _, valid_index in iterator:
        x, mask, y = prepare_data([data[0][t] for t in valid_index],
                                  numpy.array(data[1])[valid_index],
                                  maxlen=options['max_len'])
        preds = f_pred_prob(x, mask)
        targets = numpy.array(data[1])[valid_index]
        preds = preds[:, targets].mean(axis=1)
        mse_error += ((preds - 1) ** 2).sum()

    mse_error = numpy_floatX(mse_error) / len(data[0])
    return mse_error
