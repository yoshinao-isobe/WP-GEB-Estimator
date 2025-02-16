# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# evaluating test-errors with random perturbations on weights

import numpy as np
import tensorflow as tf
import utils as utl

BN_name = 'BatchNormalization'


# ------------------------------------------------
# average of empirical error by dataset with noise
# ------------------------------------------------

def measure_error_random(
        model,
        in_out_datasets,
        datasize,
        perturb_ratio,
        perturb_sample_size,
        # perturb_bn,
        verbose_measure=1,
        perturb_mode=0):

    if perturb_ratio == 0:
        err, errors = error_evaluation_batch(
            model, in_out_datasets)
        err_count = errors * perturb_sample_size

        return err_count, 0

    # empirical error with noise (Monte Carlo)

    layers = model.layers
    org_weight = [lyr.get_weights() for lyr in layers]

    # rand_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1.0)
    rand_opt = tf.keras.optimizers.SGD(learning_rate=1.0)
    params = model.trainable_variables
    perturb_params_size = utl.params_size(params)

    if perturb_mode >= 100:
        alphas = [perturb_ratio * np.mean(np.abs(params[i])) for i in range(len(params))]
    else:
        alphas = [perturb_ratio * (np.abs(params[i])) for i in range(len(params))]

    err_count = np.array([0] * datasize)
    err_sum = 0
    for k in range(perturb_sample_size):

        perturb_list = []
        for i in range(len(params)):
            alpha = alphas[i]
            param = params[i]
            rand = np.random.rand(*param.shape) * 2 - 1
            perturb = (rand * alpha).astype(np.float32)
            perturb_list.append(perturb)

        perturb_params = zip(perturb_list, params)
        rand_opt.apply_gradients(perturb_params)

        err, errors = error_evaluation_batch(
            model, in_out_datasets)

        err_count = err_count + errors

        err_sum += err
        if verbose_measure == 1:
            print('\r' + '  Random Sampling ({:d}/{:d}): an error = {:.2f} %  '.format(
                k + 1, perturb_sample_size, err * 100), end='')

        # restore weights (including biases)
        for j in range(len(layers)):
            layers[j].set_weights(org_weight[j])

    if verbose_measure == 1:
        print('\n  Average error = {:.2f} %'.format(err_sum / perturb_sample_size * 100))

    return err_count, perturb_params_size


def reshape_out_dataset(out_dataset):
    if len(out_dataset) == 0:
        return out_dataset
    else:
        if type(out_dataset[0]) is np.ndarray:
            return out_dataset.flatten()
        else:
            return out_dataset


def error_evaluation_batch(model, in_out_datasets):

    init_flag = True
    datasize = 0
    # loop_size = len(in_out_datasets)
    # loop = 1

    for (in_dataset, out_dataset) in in_out_datasets:

        datasize += len(out_dataset)
        out_dataset = reshape_out_dataset(out_dataset)

        predictions = model.predict(in_dataset, verbose=0)
        pre_labels = np.argmax(predictions, axis=1)
        errors = (pre_labels != out_dataset).astype(int)

        if init_flag:
            concat_errors = errors
            init_flag = False
        else:
            concat_errors = np.concatenate([concat_errors, errors])

    err = np.sum(concat_errors) / datasize

    return err, concat_errors


