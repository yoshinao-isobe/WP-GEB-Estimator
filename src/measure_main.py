# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# evaluating test-errors with random perturbations on weights

import math
import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import measure_params as prm

import measure_sub as msr
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

    search_out_file = root_result_dir + '/' + params.search_file + '_out.csv'
    search_id_file = root_result_dir + '/' + params.search_file + '_id.csv'

    measure_info_file = root_result_dir + '/' + params.measure_file + '_info.txt'
    measure_out_file = root_result_dir + '/' + params.measure_file + '_out.csv'

    top_line = '\n------------------------------\n'
    options = top_line + params.model_params()
    utl.save_message(measure_info_file, options, 'a')
    print(options)

    # ------------------------------------
    #  initialize measure_out
    # ------------------------------------

    search_out_list = utl.load_list(search_out_file)
    search_out_head = search_out_list[0]
    search_out_list.pop(0)

    if not utl.check_exist(measure_out_file):
        # measure csv

        str_csv = utl.list_to_str(search_out_head, ',') + ','
        str_csv += 'rnd_seed_measure,'
        str_csv += 'batch_size_measure,'
        str_csv += 'err_thr,'
        str_csv += 'err_thr_practical,'
        str_csv += 'delta,'
        str_csv += 'delta0_ratio,'
        str_csv += 'perturb_sample_size,'
        str_csv += 'err_num_random,err_num,'
        str_csv += 'test_err_wst,test_err_avr\n'

        utl.save_message(measure_out_file, str_csv, 'w')
        idx_st = 0

    else:
        measure_out_dict_list = utl.load_csv_dict_list(measure_out_file)
        idx_st = len(measure_out_dict_list)

    # ----------------------------------------------
    #  load search_out and search_id
    # ----------------------------------------------

    search_out_dict_list = utl.load_csv_dict_list(search_out_file)
    search_out_size = len(search_out_dict_list)

    err_id_array = utl.load_num_list(search_id_file)

    info_str = '\nMeasuring test error with random perturbations:\n\n'
    utl.save_message(measure_info_file, info_str, 'a')
    print(info_str)

    for idx in range(idx_st, search_out_size):

        search_out_dict = search_out_dict_list[idx]

        # ------------------------------------
        # read parameters
        # ------------------------------------

        dataset_name = search_out_dict['dataset_name']
        dataset_size = int(search_out_dict['dataset_size'])
        dataset_offset = int(search_out_dict['dataset_offset'])
        dataset_file = search_out_dict['dataset_file']
        dataset_fmt = search_out_dict['dataset_fmt']
        image_width_str = search_out_dict['image_width']
        if image_width_str == NA:
            image_width = 0
        else:
            image_width = int(image_width_str)
        image_height_str = search_out_dict['image_height']
        if image_height_str == NA:
            image_height = 0
        else:
            image_height = int(image_height_str)

        model_dir = search_out_dict['model_dir']
        perturb_ratio = float(search_out_dict['perturb_ratio'])
        perturb_bn = int(search_out_dict['perturb_bn'])
        err_num_search = int(search_out_dict['err_num_search'])

        info_str = '\nPerturbation ratio = {}'.format(perturb_ratio)
        utl.save_message(measure_info_file, info_str + '\n', 'a')
        print(info_str)

        # -----------------------------------------
        # there is no non-detected data by search
        # -----------------------------------------
        if err_num_search == dataset_size:
            info_str = '\nAll data have been detected as unsafe by search'
            utl.save_message(measure_info_file, info_str + '\n', 'a')
            print(info_str)

            # save csv file
            val_csv = utl.list_to_str(search_out_list[idx], ',') + ','
            val_csv += make_csv_str(
                params, 0, 0,0, err_num_search, 0, 0)
            utl.save_message(measure_out_file, val_csv, 'a')

        # -----------------------------------------
        # there is non-detected data by search
        # -----------------------------------------
        else:

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

            if len(err_ids) > 0:
                in_dataset = np.delete(in_dataset, np.array(err_ids), axis=0)
                out_dataset = np.delete(out_dataset, np.array(err_ids))

            # non_det_num
            dataset_size = out_dataset.shape[0]

            b_size = params.batch_size
            if b_size == 0:
                b_size = dataset_size

            in_out_datasets = [
                (in_dataset[i: i + b_size], out_dataset[i: i + b_size])
                for i in range(0, dataset_size, b_size)]

            # ------------------------------------
            # load model
            # ------------------------------------

            model = mdl.load_model(model_dir, root_result_dir)
            params_size = utl.model_trainable_params_size(model)

            info_str = 'Trainable parameters: ' + str(params_size) + '\n'

            if perturb_bn == 0:
                info_str += 'The parameters (scale, shift) in batch normalization layers are not perturbed.\n'
                model = utl.set_non_trainable_layer(model, BN_name)

            p_params_size = utl.model_trainable_params_size(model)
            info_str += 'Perturbed parameters : ' + str(p_params_size)

            utl.save_message(measure_info_file, info_str + '\n', 'a')
            print(info_str)

            # ----------------------------------------------
            #  randomly generate weight-perturbations
            # ----------------------------------------------

            # ---------------------
            # perturb sample size
            # ---------------------

            delta0 = params.delta * params.delta0_ratio
            delta1 = delta0 / dataset_size
            if params.perturb_sample_size == 0:
                perturb_sample_size = math.ceil(math.log(delta1, 1 - params.err_thr))
            else:
                perturb_sample_size = params.perturb_sample_size
            err_thr_practical = 1 - math.exp(-math.log(1 / delta1) / perturb_sample_size)

            info_str = '\nPerturbation ratio = {}'.format(perturb_ratio)
            info_str += '\n Perturbation sample size = {}'.format(perturb_sample_size)
            info_str += '\n Practical error threshold = {:.4f}% (Ideal = {:.4f}%)\n'.format(
                err_thr_practical * 100, params.err_thr * 100)
            utl.save_message(measure_info_file, info_str + '\n', 'a')
            print(info_str)

            time1 = time.time()

            err_count, perturb_params_size = msr.measure_error_random(
                model,
                in_out_datasets,
                dataset_size,
                perturb_ratio,
                perturb_sample_size,
                # params.perturb_bn,
                verbose_measure=params.verbose_measure,
                perturb_mode=0)

            time2 = time.time()
            e_time = time2 - time1

            test_err_avr = np.sum(err_count) / (dataset_size * perturb_sample_size)

            err_flag = (err_count > 0).astype(int)
            err_num_random = np.sum(err_flag)
            test_err_wst = err_num_random / dataset_size

            err_num = err_num_search + err_num_random

            # err_id_list = [i for i in range(dataset_size) if err_count[i] > 0]

            info_str = ''
            info_str += '  The number of perturbed parameters: {:d}\n'.format(perturb_params_size)
            info_str += '  The number of data whose adversarial perturbations are found by random samples:'
            info_str += ' {:d}/{:d}\n'.format(err_num_random, dataset_size)
            info_str += '  The ratio of data whose adversarial perturbations are found by random samples:'
            info_str += ' {:.2f}%\n'.format(test_err_wst * 100)
            info_str += '  The average of test-errors with random perturbations: {:.2f}%\n'.format(test_err_avr * 100)
            info_str += '  (Elapsed Time: {:.1f} [sec])\n'.format(e_time)

            utl.save_message(measure_info_file, info_str + '\n', 'a')
            print(info_str)

            # save csv file
            val_csv = utl.list_to_str(search_out_list[idx], ',') + ','
            val_csv += make_csv_str(
                params, perturb_sample_size, err_thr_practical, err_num_random, err_num, test_err_wst, test_err_avr)
            utl.save_message(measure_out_file, val_csv, 'a')

    return


def make_csv_str(
        params, perturb_sample_size, err_thr_practical, err_num_random, err_num, test_err_wst, test_err_avr):
    val_csv = str(params.random_seed) + ','
    val_csv += str(params.batch_size) + ','
    val_csv += str(params.err_thr) + ','
    val_csv += str(err_thr_practical) + ','
    val_csv += str(params.delta) + ','
    val_csv += str(params.delta0_ratio) + ','
    val_csv += str(perturb_sample_size) + ','  # perturb_sample_size
    val_csv += str(err_num_random) + ','
    val_csv += str(err_num) + ','
    val_csv += str(test_err_wst) + ','
    val_csv += str(test_err_avr) + '\n'
    return val_csv


if __name__ == '__main__':
    prm.define_default_parameters(flags)
    app.run(main)
