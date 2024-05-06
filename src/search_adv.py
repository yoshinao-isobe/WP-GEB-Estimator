# 2024/03/29, AIST
# searching for adversarial perturbations

import copy
import numpy as np
import tensorflow as tf
import utils as utl

BN_name = 'BatchNormalization'

# experiment for comparing with Tsai
# Alpha_layer = 1  # usually 0 (if 1 then the mean of each layer is used)


# ------------------------------------------------
#    wight perturbation
# ------------------------------------------------

def search_adv_perturb(
        model,
        in_out_datasets,
        search_mode,
        perturb_ratio,
        max_iteration,
        perturb_bn,
        verbose_search=1):

    # search_mode
    #  0 or 100: FGSM (FGSM on Weights)
    #  1 or 101: I-FGSM (Iterative FSGM on Weights with early-stopping)

    grad_params = model.trainable_variables
    # remove params in batch normalization layers
    if perturb_bn == 0:
        pop_lst = utl.pop_param_ids(model.layers, BN_name)
        for i in pop_lst:
            grad_params.pop(i)

    if search_mode % 100 == 0:
        err_id_list = eval_err_fgsm_main(
            model, in_out_datasets, search_mode, perturb_ratio, grad_params, verbose_search)

    elif search_mode % 100 == 1:
        err_id_list = eval_err_ifgsm_main(
            model, in_out_datasets, search_mode, perturb_ratio,
            max_iteration, grad_params, verbose_search)

    else:
        err_id_list = []
        print('perturb mode {} is not available.'.format(search_mode))

    return err_id_list


# ===================================================
#    FGSM (Fast Gradient-Sign Method on Weights)
# ===================================================

def eval_err_fgsm_main(
        model, in_out_datasets, search_mode, perturb_ratio, grad_params, verbose_search):

    offset = 0
    err_id_list = []
    for (in_dataset, out_dataset) in in_out_datasets:

        if verbose_search == 1:
            tf.print(' {:d}: '.format(offset), end='')

        sub_err_ids = eval_err_fgsm_sub(
            model, in_dataset, out_dataset, search_mode, perturb_ratio, grad_params, verbose_search)

        err_id_list += [sub_err_id + offset for sub_err_id in sub_err_ids]
        offset += len(out_dataset)

    return err_id_list


def eval_err_fgsm_sub(
        model, in_dataset, out_dataset, search_mode, perturb_ratio, grad_params, verbose_search):

    gradients = pre_example_gradients(model, in_dataset, out_dataset, grad_params)

    for i in range(len(gradients)):
        if search_mode >= 100:
            gradients[i] = - perturb_ratio * np.mean(np.abs(grad_params[i])) * np.sign(gradients[i])
        else:
            gradients[i] = - perturb_ratio * np.abs(grad_params[i]) * np.sign(gradients[i])

    err_id_list = eval_sum_err_grad_list(model, in_dataset, out_dataset, gradients, grad_params, verbose_search)
    return err_id_list


def eval_sum_err_grad_list(model, in_dataset, out_dataset, gradients, grad_params, verbose_search):
    layers = model.layers
    org_weight = [lyr.get_weights() for lyr in layers]

    err_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    data_size = len(in_dataset)
    grad_size = len(gradients)
    gradients_list = [list(gradients[i]) for i in range(grad_size)]

    if verbose_search == 1:
        tf.print(' [', end='')
    bar = data_size // 50
    if bar == 0:
        bar = 1

    err_id_list = []
    for i in range(data_size-1, -1, -1):
        if verbose_search == 1:
            if i % bar == 0:
                tf.print('=', end='')

        grads = [gradients_list[j].pop() for j in range(grad_size)]
        grads_and_vars = zip(grads, grad_params)
        err_opt.apply_gradients(grads_and_vars)

        err = eval_single_error(model, in_dataset[i], out_dataset[i])
        if err == 1:
            err_id_list += [i]

        for j in range(len(layers)):
            layers[j].set_weights(org_weight[j])
    if verbose_search == 1:
        tf.print(']')

    err_id_list.sort()
    # tf.print('err_ids = ', err_ids)

    return err_id_list


# ===================================================
#    1: I-FGSM (Iterative FGSM with early-stopping)
# ===================================================

def eval_err_ifgsm_main(
        model, in_out_datasets, search_mode, perturb_ratio,
        max_iteration, grad_params, verbose_search):

    layers = model.layers
    org_weight = [lyr.get_weights() for lyr in layers]

    in_dataset, out_dataset = utl.flatten_batches(in_out_datasets)
    data_size = len(out_dataset)

    err_id_list = []
    for i in range(data_size):

        if verbose_search == 1:
            if i % 20 == 0:
                tf.print('{:8d}: '.format(i), end='')

        in_data = in_dataset[i]
        out_data = out_dataset[i]

        err = eval_err_ifgsm_sub(
            model, in_data, out_data, search_mode, perturb_ratio,
            max_iteration, grad_params, verbose_search)
        if err == 1:
            err_id_list.append(i)

        for j in range(len(layers)):
            layers[j].set_weights(org_weight[j])

        if verbose_search == 1:
            if (i + 1) % 20 == 0:
                tf.print()

    if verbose_search == 1:
        tf.print()

    err_id_list.sort()
    # tf.print('err_ids = ', err_ids)

    # avg_err = sum_err / data_size
    return err_id_list


def eval_err_ifgsm_sub(
        model, in_data, out_data, search_mode, perturb,
        max_iteration, grad_params, verbose_search):

    err = eval_single_error(model, in_data, out_data)
    if err == 1:
        if verbose_search == 1:
            tf.print(' 0x', end='')
        return err

    mgn = 0
    if search_mode % 100 == 1:
        mgn = eval_single_margin(model, in_data, out_data)

    err_opt = tf.keras.optimizers.legacy.SGD(learning_rate=1.0)

    # params = model.trainable_variables
    org_grad_params = copy.deepcopy(grad_params)

    if search_mode >= 100:
        alpha = [perturb * np.mean(np.abs(org_grad_params[i])) for i in range(len(org_grad_params))]
    else:
        alpha = [perturb * (np.abs(org_grad_params[i])) for i in range(len(org_grad_params))]

    for epoch in range(max_iteration):

        grads = single_gradients(model, in_data, out_data, grad_params)

        # I-FGSM (iterative)
        if search_mode % 100 == 1:
            for i in range(len(grads)):
                sg = np.sign(grads[i])
                g = - alpha[i] * sg
                grads[i] = grad_params[i] - (org_grad_params[i] - g)

        # update weights
        grads_and_vars = zip(grads, grad_params)
        err_opt.apply_gradients(grads_and_vars)

        err = eval_single_error(model, in_data, out_data)
        if err == 1:
            if verbose_search == 1:
                tf.print('{:2d}x'.format(epoch+1), end='')
            return 1

        if search_mode % 100 == 1:
            new_mgn = eval_single_margin(model, in_data, out_data)
            if new_mgn >= mgn:
                if verbose_search == 1:
                    tf.print('{:2d} '.format(epoch + 1), end='')
                return 0
            mgn = new_mgn

    if verbose_search == 1:
        tf.print('-- ', end='')
    return 0


# ===========================
#     Common functions
# ===========================

# --------------------
#  Single error
# --------------------

def eval_single_error(model, idata, odata):
    tf_idata = tf.expand_dims(idata, 0)
    predict = model(tf_idata)
    predicted_class = tf.cast(tf.argmax(predict, axis=1), tf.uint8)
    err1 = tf.cast(tf.math.not_equal(odata, predicted_class), tf.int32)
    err = err1[0]
    return err


# --------------------
#    gradients
# --------------------

@tf.function(reduce_retracing=True)
def single_gradients(model, in_data, out_data, params):
    loss_fun = tf.keras.losses.SparseCategoricalCrossentropy()
    in_data0 = tf.expand_dims(in_data, 0)
    out_data0 = tf.expand_dims(out_data, 0)

    with tf.GradientTape() as tape:
        tape.watch(params)
        out_predict1 = model(in_data0)
        neg_loss = loss_fun(out_data0, out_predict1)

    grad = tape.gradient(neg_loss, params)

    return grad


# if perturb_bn is 0 then the gradients in layer "name_BN" is 0.
@tf.function(reduce_retracing=True)
def pre_example_gradients(model, in_dataset, out_dataset, params):

    def pre_single_gradients(in_out_data):
        in_data, out_data = in_out_data
        grad = single_gradients(model, in_data, out_data, params)
        return grad

    gradients = tf.vectorized_map(pre_single_gradients, (in_dataset, out_dataset))
    return gradients


# --------------------
#  Single margin
# --------------------

def eval_single_margin(model, idata, odata):

    tf_idata = tf.expand_dims(idata, 0)
    # print(' odata = ', odata)
    predict = model(tf_idata)
    predict0 = predict[0]
    # tf.print(' predict.numpy = ', predict.numpy())
    # tf.print(' type(odata) = ', type(odata))
    if type(odata) is np.ndarray:
        odata = odata[0]
    correct_prob = predict0[odata]
    probs, indices = tf.math.top_k(predict0, k=2)
    if correct_prob == probs[0]:
        other_prob = probs[1]
    else:
        other_prob = probs[0]

    margin = correct_prob - other_prob
    # margin = np.log(correct_prob) - np.log(other_prob)

    # return margin
    return margin.numpy()


