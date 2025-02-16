# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# searching for adversarial perturbations by weight-gradients

import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import search_params as prm

import search_sub as sch
import utils as utl
import dataset as dst
import model as mdl

NA = 'N/A'
Results_dir = 'results'
BN_name = 'BatchNormalization'


def main(args):

    # ------------------------------------
    # set parameters
    # ------------------------------------

    params = prm.OptParams(flags.FLAGS)
    if params.random_seed != 0:
        np.random.seed(params.random_seed)
        tf.random.set_seed(params.random_seed)

    # mkdir
    root_result_dir = Results_dir + '/' + params.result_dir
    utl.chk_mkdir(root_result_dir)

    search_info_file = root_result_dir + '/' + params.search_file + '_info.txt'
    search_out_file = root_result_dir + '/' + params.search_file + '_out.csv'
    search_id_file = root_result_dir + '/' + params.search_file + '_id.csv'

    top_line = '\n------------------------------\n'
    options = top_line + params.model_params()
    utl.save_message(search_info_file, options, 'a')
    print(options)

    # ------------------------------------
    # load dataset
    # ------------------------------------

    image_width = params.image_width
    image_height = params.image_height
    if params.dataset_name == 'imagenet' and image_width == 0 and image_height == 0:
        image_width, image_height = mdl.set_image_size(params.model_dir)

    dataset = dst.Dataset()
    dataset.load_dataset(
        params.dataset_name,
        params.dataset_size,
        params.dataset_offset,
        dataset_file=params.dataset_file,
        dataset_fmt=params.dataset_fmt,
        image_width=image_width,
        image_height=image_height,
        model_dir=params.model_dir,
        train_flag=False)

    if params.batch_size == 0:
        params.batch_size = dataset.dataset_size
    batch_size = params.batch_size

    in_dataset = dataset.in_dataset
    out_dataset = dataset.out_dataset
    # dataset_size = dataset.dataset_size     # in_dataset.shape[0]

    in_out_datasets = [
        (in_dataset[i: i + batch_size], out_dataset[i: i + batch_size])
        for i in range(0, params.dataset_size, batch_size)]

    # ------------------------------------
    # load model
    # ------------------------------------

    model = mdl.load_model(params.model_dir, root_result_dir)

    params_size = utl.model_trainable_params_size(model)
    info_str = '\nMeasuring test error with random perturbations:\n\n'
    info_str += 'Trainable parameters: ' + str(params_size) + '\n'

    if params.perturb_bn == 0:
        info_str += 'The parameters (scale, shift) in batch normalization layers are not perturbed.\n'
        model = utl.set_non_trainable_layer(model, BN_name)

    p_params_size = utl.model_trainable_params_size(model)
    info_str += 'Perturbed parameters : ' + str(p_params_size)

    print(info_str)
    utl.save_message(search_info_file, info_str + '\n', 'a')

    # ------------------------------------
    #  initialize search_out
    # ------------------------------------

    if not utl.check_exist(search_out_file):
        # csv measure

        str_csv = 'dataset_name,dataset_size,dataset_offset,'
        str_csv += 'dataset_file,dataset_fmt,image_width,image_height,'
        str_csv += 'model_dir,'
        str_csv += 'rnd_seed_search,'
        str_csv += 'batch_size_search,'
        str_csv += 'perturb_bn,'
        str_csv += 'perturb_ratio,'
        str_csv += 'search_mode,'
        str_csv += 'max_iteration,err_num_search\n'

        utl.save_message(search_out_file, str_csv, 'w')
        utl.save_message(search_id_file, '', 'w')

        val_csv = make_csv_str(params, image_width, image_height, 0, 0)
        utl.save_message(search_out_file, val_csv, 'a')

        # save err id
        utl.save_message(search_id_file, '\n', 'a')

    # ----------------------------------------------
    #  load measure_out and measure_id
    # ----------------------------------------------

    info_str = '\nSearching adversarial perturbations:\n\n'
    utl.save_message(search_info_file, info_str, 'a')
    print(info_str)

    for i in range(len(params.perturb_ratios)):

        perturb_ratio = params.perturb_ratios[i]
        info_str = '\nPerturbation ratio = {}'.format(perturb_ratio)
        utl.save_message(search_info_file, info_str + '\n', 'a')
        print(info_str)

        # -----------------------------------------------------
        #  searching adversarial perturbation by FGSM / I-FGSM
        # -----------------------------------------------------

        if params.skip_search == 0:

            time1 = time.time()

            if perturb_ratio == 0 or params.dataset_size == 0:
                err_id_list = []
            else:
                err_id_list = sch.search_adv_perturb(
                    model,
                    in_out_datasets,
                    params.search_mode,
                    perturb_ratio,
                    params.max_iteration,
                    # perturb_bn,
                    verbose_search=params.verbose_search)
            err_num_search = len(err_id_list)

            time2 = time.time()

            info_str = ''
            info_str += '  The number of data whose adversarial perturbations are found:\n'
            info_str += '    by gradient-search: {:d}/{:d}\n'.format(
                err_num_search, params.dataset_size)
            info_str += '  (Elapsed Time: {:.1f} [sec])'.format(time2 - time1)

            utl.save_message(search_info_file, info_str + '\n', 'a')
            print(info_str)

        else:
            err_num_search = 0
            err_id_list = []
            # err_num = 0
            info_str = '  The search for adversarial perturbations is skipped.'
            utl.save_message(search_info_file, info_str + '\n', 'a')
            print(info_str)

        # save csv file
        val_csv = make_csv_str(params, image_width, image_height, perturb_ratio, err_num_search)
        utl.save_message(search_out_file, val_csv, 'a')

        err_id_str = utl.list_to_str(err_id_list, ',')
        utl.save_message(search_id_file, err_id_str + '\n', 'a')

    return


def make_csv_str(params, image_width, image_height, perturb_ratio, err_num_search):
    val_csv = str(params.dataset_name) + ','
    val_csv += str(params.dataset_size) + ','
    val_csv += str(params.dataset_offset) + ','
    if params.dataset_name in dst.Keras_dataset:
        val_csv += NA + ',' + NA + ',' + NA + ',' + NA + ','
    else:
        val_csv += params.dataset_file + ','
        val_csv += str(params.dataset_fmt) + ','
        val_csv += str(image_width) + ','
        val_csv += str(image_height) + ','
    val_csv += str(params.model_dir) + ','

    val_csv += str(params.random_seed) + ','
    val_csv += str(params.batch_size) + ','
    val_csv += str(params.perturb_bn) + ','
    val_csv += str(perturb_ratio) + ','

    if params.skip_search == 1:
        val_csv += NA + ',' + NA + ', 0\n'
    else:
        val_csv += str(params.search_mode) + ','
        val_csv += str(params.max_iteration) + ','
        val_csv += str(err_num_search) + '\n'

    return val_csv


if __name__ == '__main__':
    prm.define_default_parameters(flags)
    app.run(main)
