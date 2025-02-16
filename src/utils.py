# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# utilities

import os
import math
import numpy as np
import csv
import tensorflow as tf


len_name = 20
root_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'

MODEL_EXT = '.keras'


# ---------------------------------
#  basic functions
# ---------------------------------

def flag_to_idx(err_flag):
    if type(err_flag) is np.ndarray:
        size = err_flag.shape[0]
    else:
        size = len(err_flag)
    err_idx = [i for i in range(size) if err_flag[i] == 1]
    return err_idx


def complement_list(lst, lst_size):
    c_lst = [i for i in range(lst_size) if i not in lst]
    return c_lst


#  ------------------------------------------------------
#   make id list for popping parameters of removed layers
#  ------------------------------------------------------

def pop_param_ids(layers, remove_layer_name):

    ids = []
    k = 0
    for i in range(len(layers)):
        layer = layers[i]
        weights = layer.trainable_weights
        for j in range(len(weights)):
            if layer.__class__.__name__ == remove_layer_name:
                ids.append(k)
            else:
                k += 1
    return ids


# perturbed params ids
def param_ids(layers, remove_layer_name):
    ids = []
    k = 0
    for i in range(len(layers)):
        layer = layers[i]
        weights = layer.trainable_weights
        for j in range(len(weights)):
            if layer.__class__.__name__ != remove_layer_name\
                    or remove_layer_name == '':
                ids.append(k)
            k += 1
    return ids

# ---------------------------------
# flatten batches
# ---------------------------------

def flatten_batches(pair_datasets):
    init_flag = True
    for (ds1, ds2) in pair_datasets:
        if init_flag:
            dataset1 = ds1
            dataset2 = ds2
            init_flag = False
        else:
            # dataset1 = dataset1 + ds1
            # dataset2 = dataset2 + ds2
            dataset1 = np.concatenate([dataset1, ds1], axis=0)
            dataset2 = np.concatenate([dataset2, ds2], axis=0)
    return dataset1, dataset2


# ---------------------------------
# evaluation by batches
# ---------------------------------

def evaluate_batch(model, in_out_datasets, verbose=0):
    corrects = 0
    size = 0

    for (in_dataset, out_dataset) in in_out_datasets:
        size += in_dataset.shape[0]
        c = evaluate_corrects(
            model,
            in_dataset,
            out_dataset,
            verbose=verbose)
        corrects += c
        # print('*** in_dataset.shape[0] = ', in_dataset.shape[0])
        # print('*** c = ', c)

    acc = corrects / size
    err = (size - corrects) / size
    print('err = ', err)
    return acc, err


def evaluate_corrects(model, in_dataset, out_dataset, verbose=0):

    p_yxs = model.predict(in_dataset, verbose=verbose)
    p_labels = np.argmax(p_yxs, axis=1)
    # print('out_dataset = ', out_dataset)
    # print('p_labels = ', p_labels)
    corrects_vec = (out_dataset == p_labels).astype(int)
    corrects = sum(corrects_vec)
    return corrects


# ---------------------------------
# numerical computation
# ---------------------------------

# kl^(-1)(q,c)
def inv_binary_kl_div(q, c, eps_nm, max_nm):
    if q == 0:
        p = 1.0 - math.exp(-c)
        return to_float(p)

    else:
        p = q + math.sqrt(c / 2.0)

        for i in range(max_nm):
            if p >= 1.0:
                return 1.0

            if math.fabs(p - c) < eps_nm:
                return to_float(p)

            h1 = binary_kl_div(q, p) - c
            h2 = (1 - q) / (1 - p) - q / p
            p = p - h1 / h2

        return to_float(p)


def binary_kl_div(q, p):
    kl = q * math.log(q / p) + (1 - q) * math.log((1 - q) / (1 - p))

    return kl


def to_float(p):
    if p.__class__.__name__ == 'EagerTensor':
        return p.numpy()
    else:
        return p


def model_params_size(model):
    tr_size = model_trainable_params_size(model)
    non_tr_size = params_size(model.non_trainable_weights)
    total_size = int(tr_size + non_tr_size)
    return total_size


def model_trainable_params_size(model):
    return params_size(model.trainable_weights)


def params_size(params):
    # tr_size = np.sum([np.prod(v.get_shape()) for v in params])
    tr_size = np.sum([np.prod(v.shape) for v in params])
    total_size = int(tr_size)
    return total_size

# ---------------------------------
# load/save model-files
# ---------------------------------


def load_model(fn):
    # load the trained model

    dir_fn = root_dir + fn + MODEL_EXT
    # dir_fn = root_dir + fn + '.h5'
    # dir_fn = root_dir + fn
    print('load the model from: ...', dir_fn[-1 * len_name:])
    model = tf.keras.models.load_model(dir_fn)
    # model.summary()
    return model


def save_model(fn, model):
    # save the trained model
    # root_dir = os.path.dirname(os.path.realpath(__file__)) + '/../'
    dir_fn = root_dir + fn + MODEL_EXT
    print('save the model to: ...', dir_fn[-1 * len_name:])
    model.save(dir_fn)


# ---------------------------------
# load/save csv-files
# ---------------------------------

def history_to_csv_str(history):
    s = ''
    k_list = list(history.keys())
    k_size = len(k_list)
    k0 = k_list[0]
    list0 = history[k0]
    d_size = len(list0)

    # header
    for i in range(k_size):
        s += k_list[i]
        if i == k_size - 1:
            s += '\n'
        else:
            s += ', '
    # data
    for j in range(d_size):
        for i in range(k_size):
            ds = history[k_list[i]]
            s += str(ds[j])
            if i == k_size - 1:
                s += '\n'
            else:
                s += ', '
    return s


# ---------------------------------
# make dir
# ---------------------------------

def chk_mkdir(new_dir):
    root_new_dir = root_dir + new_dir
    if not os.path.isdir(root_new_dir):
        os.mkdir(root_new_dir)
        print('make: ', root_new_dir)


def save_message(fn, message, mode):  # mode = 'w' or 'a'
    fn = root_dir + fn
    f = open(fn, mode)
    f.write(message)
    f.close()


def check_exist(fn):
    dir_fn = root_dir + fn
    return os.path.isfile(dir_fn)


def load_list(fn):
    fn = root_dir + fn
    f = open(fn, 'r')
    reader = csv.reader(f)
    row_list = []
    for row in reader:
        row_list.append(row)
    return row_list


def load_num_list(fn, float_flag=False):
    row_list = load_list(fn)
    num_array = []
    for row in row_list:
        if float_flag:
            num_list = [float(s) for s in row if s != '']  # .isdecimal()]
        else:
            num_list = [int(s) for s in row if s != '']  # .isdecimal()]

        num_array.append(num_list)
    return num_array


def load_csv_dict_list(fn):
    fn = root_dir + fn  # + '.csv'
    f = open(fn, 'r')
    dic = csv.DictReader(f, delimiter=',')

    dict_list = []
    for row in dic:
        dict_list.append(row)

    f.close()

    # print('dict_list = ', dict_list)
    return dict_list


# string '(1,2)' --> (1,2)
def str_to_int_tuple(tuple_str):
    tuple_str = tuple_str.replace('(', '')
    tuple_str = tuple_str.replace(')', '')
    list_str = tuple_str.split(',')

    # print('list_str = ', list_str)

    int_tuple = ()
    for s in list_str:
        int_tuple = int_tuple + (int(s),)

    return int_tuple


def dict_list_str(dict_list):
    st = ''
    i = 0
    for d in dict_list:
        st += '{:3d}. '.format(i)
        i += 1
        key_list = list(d.keys())

        for j in range(len(key_list)):
            k = key_list[j]
            if d[k] != '':
                st += '{:s}:'.format(k) + str(d[k])
                if i != len(key_list) - 1:
                    st += ', '
        st += '\n'
    return st


# [1,2,3] -> '1, 2, 3'
def list_to_str(ls, delimiter):
    ls_str = ''
    for j in range(len(ls)):
        ls_str += str(ls[j])
        if j < len(ls) - 1:
            ls_str += delimiter
    return ls_str


# ------------------------------------------------------------
#   set trainable attribute to non_trainable in layer_name
# ------------------------------------------------------------

def set_non_trainable_layer(model, layer_name):
    layers = model.layers
    for i in range(len(layers)):
        layer = layers[i]
        weights = layer.trainable_weights
        for j in range(len(weights)):
            if layer.__class__.__name__ == layer_name:
                layer.trainable = False
    return model
