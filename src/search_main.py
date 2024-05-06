# Copyright (C) 2024
# National Institute of Advanced Industrial Science and Technology (AIST)

# searching for adversarial perturbations by wight-gradients

import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import search_params as prm

import search_adv as sch
import utils as utl
import dataset as dst
import model as mdl

NA = 'N/A'
Results_dir = 'results'


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

    measure_out_file = root_result_dir + '/' + params.measure_file + '_out.csv'
    measure_id_file = root_result_dir + '/' + params.measure_file + '_id.csv'

    options = params.model_params()
    utl.save_message(search_info_file, options, 'w')
    print(options)

    # ------------------------------------
    #  initialize search_out
    # ------------------------------------

    measure_out_list = utl.load_list(measure_out_file)
    measure_out_head = measure_out_list[0]
    measure_out_list.pop(0)

    if not utl.check_exist(search_out_file):
        # search csv

        str_csv = utl.list_to_str(measure_out_head, ',') + ','
        str_csv += 'rnd_seed_search,'
        str_csv += 'batch_size_search,'
        str_csv += 'search_mode,'
        str_csv += 'max_iteration,err_num_search,err_num\n'

        utl.save_message(search_out_file, str_csv, 'w')
        idx_st = 0

    else:
        search_out_dict_list = utl.load_csv_dict_list(search_out_file)
        idx_st = len(search_out_dict_list)

    # ----------------------------------------------
    #  load measure_out and measure_id
    # ----------------------------------------------

    measure_out_dict_list = utl.load_csv_dict_list(measure_out_file)
    measure_out_size = len(measure_out_dict_list)

    err_id_array = utl.load_num_list(measure_id_file)

    info_str = '\nSearching adversarial perturbations:\n\n'
    utl.save_message(search_info_file, info_str, 'a')
    print(info_str)

    for idx in range(idx_st, measure_out_size):

        measure_out_dict = measure_out_dict_list[idx]

        dataset_name = measure_out_dict['dataset_name']
        dataset_size = int(measure_out_dict['dataset_size'])
        dataset_offset = int(measure_out_dict['dataset_offset'])
        dataset_file = measure_out_dict['dataset_file']
        dataset_fmt = measure_out_dict['dataset_fmt']
        image_width_str = measure_out_dict['image_width']
        if image_width_str == NA:
            image_width = 0
        else:
            image_width = int(image_width_str)
        image_height_str = measure_out_dict['image_height']
        if image_height_str == NA:
            image_height = 0
        else:
            image_height = int(image_height_str)

        model_dir = measure_out_dict['model_dir']
        perturb_ratio = float(measure_out_dict['perturb_ratio'])
        perturb_bn = int(measure_out_dict['perturb_bn'])
        err_num_random = int(measure_out_dict['err_num_random'])

        info_str = '\nPerturbation ratio = {}'.format(perturb_ratio)
        utl.save_message(search_info_file, info_str + '\n', 'a')
        print(info_str)

        # ------------------------------------
        # load dataset and err_id
        # ------------------------------------

        dataset = dst.Dataset()
        dataset.load_dataset(
            dataset_name=dataset_name,
            dataset_size=dataset_size,
            dataset_offset=dataset_offset,
            dataset_file=dataset_file,
            dataset_fmt=dataset_fmt,
            image_width=image_width,
            image_height=image_height,
            model_dir=model_dir,
            train_flag=False)

        in_dataset = dataset.in_dataset
        out_dataset = dataset.out_dataset

        # delete data, for which perturbations have found by (I-)FGSM
        err_ids = err_id_array[idx]

        dataset_size_all = out_dataset.shape[0]

        if len(err_ids) > 0:
            in_dataset = np.delete(in_dataset, np.array(err_ids), axis=0)
            out_dataset = np.delete(out_dataset, np.array(err_ids))

        dataset_size = out_dataset.shape[0]

        b_size = params.batch_size
        in_out_datasets = [
            (in_dataset[i: i + b_size], out_dataset[i: i + b_size])
            for i in range(0, dataset_size, b_size)]

        # ------------------------------------
        # load model
        # ------------------------------------

        model = mdl.load_model(model_dir, root_result_dir)

        # -----------------------------------------------------
        #  searching adversarial perturbation by FGSM / I-FGSM
        # -----------------------------------------------------

        if params.skip_search == 0:

            time1 = time.time()

            if perturb_ratio == 0 or dataset_size == 0:
                err_num_search = 0
            else:
                err_id_list = sch.search_adv_perturb(
                    model,
                    in_out_datasets,
                    params.search_mode,
                    perturb_ratio,
                    params.max_iteration,
                    perturb_bn,
                    verbose_search=params.verbose_search)
                err_num_search = len(err_id_list)

            time2 = time.time()

            if perturb_ratio == 0:
                search_dataset_size = dataset_size - err_num_random
            else:
                search_dataset_size = dataset_size

            err_num = err_num_random + err_num_search

            info_str = ''
            info_str += '  The number of data whose adversarial perturbations are found:\n'
            info_str += '    by gradient-search: {:d}/{:d}\n'.format(
                err_num_search, search_dataset_size)
            info_str += '    by gradient-search and random-sample: {:d}/{:d}\n'.format(
                err_num, dataset_size_all)
            info_str += '  (Elapsed Time: {:.1f} [sec])'.format(time2 - time1)

            utl.save_message(search_info_file, info_str + '\n', 'a')
            print(info_str)

        else:
            err_num_search = 0
            err_num = 0
            info_str = '  The search for adversarial perturbations is skipped.'
            utl.save_message(search_info_file, info_str + '\n', 'a')
            print(info_str)

        # save csv file
        val_csv = utl.list_to_str(measure_out_list[idx], ',') + ','
        val_csv += str(params.random_seed) + ','
        val_csv += str(params.batch_size) + ','
        if params.skip_search == 1:
            val_csv += NA + ',' + NA + ',' + NA + ',' + NA + ',' + NA + '\n'
        else:
            val_csv += str(params.search_mode) + ','
            val_csv += str(params.max_iteration) + ','
            val_csv += str(err_num_search) + ','
            val_csv += str(err_num) + '\n'
        utl.save_message(search_out_file, val_csv, 'a')

    return


if __name__ == '__main__':
    prm.define_default_parameters(flags)
    app.run(main)
