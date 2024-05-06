# Copyright (C) 2024
# National Institute of Advanced Industrial Science and Technology (AIST)

# evaluating test-errors with random perturbations on wights

import time
from absl import app
from absl import flags
import numpy as np
import tensorflow as tf

import measure_params as prm

import measure_rnd as msr
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

    measure_info_file = root_result_dir + '/' + params.measure_file + '_info.txt'
    measure_out_file = root_result_dir + '/' + params.measure_file + '_out.csv'
    measure_id_file = root_result_dir + '/' + params.measure_file + '_id.csv'

    options = params.model_params()
    utl.save_message(measure_info_file, options, 'w')
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

    in_dataset = dataset.in_dataset
    out_dataset = dataset.out_dataset
    dataset_size = dataset.dataset_size     # in_dataset.shape[0]

    batch_size = params.batch_size
    if batch_size == 0:
        batch_size = dataset_size

    in_out_datasets = [
        (in_dataset[i: i + batch_size], out_dataset[i: i + batch_size])
        for i in range(0, dataset_size, batch_size)]

    # ------------------------------------
    # load model
    # ------------------------------------

    model = mdl.load_model(params.model_dir, root_result_dir)

    params_size = utl.model_trainable_params_size(model)
    info_str = '\nMeasuring test error with random perturbations:\n\n'
    info_str += 'Trainable parameters: ' + str(params_size)
    print(info_str)
    utl.save_message(measure_info_file, info_str + '\n', 'a')

    # ------------------------------------
    #  initialize measure_out
    # ------------------------------------

    if not utl.check_exist(measure_out_file):
        # csv measure

        str_csv = 'rnd_seed_measure,dataset_name,dataset_size,dataset_offset,'
        str_csv += 'dataset_file,dataset_fmt,image_width,image_height,batch_size_measure,'
        str_csv += 'model_dir,'
        str_csv += 'perturb_bn,perturb_params_size,perturb_ratio,'

        str_csv += 'perturb_sample_size,'
        str_csv += 'err_num_random,test_err_wst,test_err_avr\n'

        utl.save_message(measure_out_file, str_csv, 'w')
        utl.save_message(measure_id_file, '', 'w')

        test_err, errors = msr.error_evaluation_batch(
            model, in_out_datasets)

        # test_err = 1.0 - acc
        info_str = 'Test error : ' + str(round(test_err * 100, 2)) + '% (no perturbation)'

        print(info_str)
        utl.save_message(measure_info_file, info_str + '\n', 'a')

        # save csv file for no perturbation
        val_csv = str(params.random_seed) + ','
        val_csv += str(params.dataset_name) + ','
        val_csv += str(dataset_size) + ','
        val_csv += str(params.dataset_offset) + ','
        if params.dataset_name in dst.Keras_dataset:
            val_csv += NA + ',' + NA + ',' + NA + ',' + NA + ','
        else:
            val_csv += params.dataset_file + ','
            val_csv += str(params.dataset_fmt) + ','
            val_csv += str(image_width) + ','
            val_csv += str(image_height) + ','
        val_csv += str(params.batch_size) + ','
        val_csv += str(params.model_dir) + ','
        val_csv += str(params.perturb_bn) + ','
        val_csv += str(0) + ','
        val_csv += str(0) + ','
        val_csv += str(params.perturb_sample_size) + ','

        val_csv += str(int(test_err * dataset_size + 0.5)) + ','
        val_csv += str(test_err) + ','
        val_csv += str(test_err) + '\n'

        utl.save_message(measure_out_file, val_csv, 'a')

        # save err id
        utl.save_message(measure_id_file, '\n', 'a')

    # -----------------------------------------------------------------
    #  measure the number of errors with random weight-perturbations
    # -----------------------------------------------------------------

    for i in range(len(params.perturb_ratios)):

        perturb_ratio = params.perturb_ratios[i]
        info_str = '\nPerturbation ratio = {}'.format(perturb_ratio)
        utl.save_message(measure_info_file, info_str + '\n', 'a')
        print(info_str)

        time1 = time.time()

        err_count, perturb_params_size = msr.measure_error_random(
            model,
            in_out_datasets,
            dataset_size,
            perturb_ratio,
            params.perturb_sample_size,
            params.perturb_bn,
            verbose_measure=params.verbose_measure,
            perturb_mode=0)

        time2 = time.time()
        e_time = time2 - time1

        test_err_avr = np.sum(err_count) / (dataset_size * params.perturb_sample_size)
        err_flag = (err_count > 0).astype(int)
        err_num = np.sum(err_flag)
        test_err_wst = err_num / dataset_size

        err_id_list = [i for i in range(dataset_size) if err_count[i] > 0]

        info_str = ''
        info_str += '  The number of perturbed parameters: {:d}\n'.format(perturb_params_size)
        info_str += '  The number of data whose adversarial perturbations are found by random samples: {:d}/{:d}\n'.format(
            err_num, dataset_size)
        info_str += '  The ratio of data whose adversarial perturbations are found by random samples: {:.2f}%\n'.format(
                test_err_wst * 100)
        info_str += '  The average of test-errors with random perturbations: {:.2f}%\n'.format(test_err_avr * 100)
        info_str += '  (Elapsed Time: {:.1f} [sec])\n'.format(e_time)

        utl.save_message(measure_info_file, info_str + '\n', 'a')
        print(info_str)

        # save csv file

        # save csv file for no perturbation
        val_csv = str(params.random_seed) + ','
        val_csv += str(params.dataset_name) + ','
        val_csv += str(dataset_size) + ','
        val_csv += str(params.dataset_offset) + ','
        if params.dataset_name in dst.Keras_dataset:
            val_csv += NA + ',' + NA + ',' + NA + ',' + NA + ','
        else:
            val_csv += params.dataset_file + ','
            val_csv += str(params.dataset_fmt) + ','
            val_csv += str(image_width) + ','
            val_csv += str(image_height) + ','
        val_csv += str(params.batch_size) + ','
        val_csv += str(params.model_dir) + ','
        val_csv += str(params.perturb_bn) + ','
        val_csv += str(perturb_params_size) + ','
        val_csv += str(perturb_ratio) + ','
        val_csv += str(params.perturb_sample_size) + ','  # perturb_sample_size

        val_csv += str(err_num) + ','
        val_csv += str(test_err_wst) + ','
        val_csv += str(test_err_avr) + '\n'

        utl.save_message(measure_out_file, val_csv, 'a')

        # save err id
        err_id_str = utl.list_to_str(err_id_list, ',')
        utl.save_message(measure_id_file, err_id_str + '\n', 'a')

    return


if __name__ == '__main__':
    prm.define_default_parameters(flags)
    app.run(main)
