# Copyright (C) 2024
# National Institute of Advanced Industrial Science and Technology (AIST)

# training models (neural classifiers) for demonstrations

from absl import app
from absl import flags
import numpy as np
import tensorflow as tf
import time

import train_params as prm

import dataset as dst
import utils as utl
import net_const as nct

Results_dir = 'results'


def main(args):
    # ------------------------------------
    # set parameters
    # ------------------------------------

    params = prm.InitParams(flags.FLAGS)
    if params.random_seed != 0:
        np.random.seed(params.random_seed)
        tf.random.set_seed(params.random_seed)

    # mkdir and file names
    rr_dir = Results_dir + '/' + params.result_dir
    utl.chk_mkdir(rr_dir)
    train_file = rr_dir + '/' + params.train_file + '_info.txt'
    train_log_file = rr_dir + '/' + params.train_file + '_log.csv'
    train_model_dir = rr_dir + '/' + params.model_dir

    # save
    options = params.model_params()
    utl.save_message(train_file, options, 'w')
    print(options)

    # ------------------------------------
    # load dataset
    # ------------------------------------

    dataset = dst.Dataset()
    dataset.load_dataset(
        dataset_name=params.dataset_name,
        dataset_size=params.train_dataset_size,
        dataset_offset=params.train_dataset_offset,
        train_flag=True)

    # separation: training dataset -> training and validation
    (in_valid_dataset, out_valid_dataset), (in_train_dataset, out_train_dataset) =\
        dataset.separate_dataset(params.validation_ratio)

    # ------------------------------------
    # training
    # ------------------------------------

    in_shape = in_train_dataset[0].shape

    net_arch_dict_list = utl.load_csv_dict_list(params.net_arch_file + '.csv')

    arch_info_str = 'Architecture:\n' + utl.dict_list_str(net_arch_dict_list)
    print(arch_info_str)
    utl.save_message(train_file, arch_info_str + '\n', 'a')

    model = nct.net_const(
        in_shape=in_shape,
        # out_size=dataset.out_size,
        net_arch_dict_list=net_arch_dict_list,
        sigma=params.sigma,
        regular_l2=params.regular_l2,
        dropout_rate=params.dropout_rate)

    model.summary()

    if params.decay_steps == 0:
        opt = tf.keras.optimizers.SGD(
            learning_rate=params.learning_rate,
            # decay=1e-6,
            momentum=0.9, nesterov=True)
    else:
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            params.learning_rate,
            decay_steps=params.decay_steps,
            decay_rate=params.decay_rate,
            staircase=True)

        opt = tf.keras.optimizers.SGD(
            # learning_rate=params.learning_rate,
            learning_rate=lr_schedule,
            # decay=1e-6,
            momentum=0.9, nesterov=True)

    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(
        optimizer=opt,  # optimizer: SGD
        loss=loss,
        metrics=['accuracy'])

    callbacks = []
    if params.early_stop == 1:
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            min_delta=params.early_stop_delta,
            patience=params.early_stop_patience)
        callbacks.append(early_stopping)

    # fitting
    time1 = time.time()
    history = model.fit(
        in_train_dataset,
        out_train_dataset,
        batch_size=params.batch_size,
        epochs=params.epochs,
        validation_data=(in_valid_dataset, out_valid_dataset),
        callbacks=callbacks,
        verbose=params.verbose)
    time2 = time.time()
    time_str = 'Fitting time: {:.1f} (sec)\n'.format(time2 - time1)
    print(time_str)
    utl.save_message(train_file, time_str + '\n', 'a')

    # ---- save logs -----

    key_list = list(history.history.keys())
    ep_size = len(history.history[key_list[0]])

    # epoch_list = list(range(params.epochs))
    epoch_list = list(range(ep_size))
    history.history = {'epoch': epoch_list} | history.history

    s = utl.history_to_csv_str(history.history)
    utl.save_message(train_log_file, s, 'w')

    # ---- save the (trained) model -----

    utl.save_model(train_model_dir, model)

    # ---- test the (trained) model -----

    loss, acc = model.evaluate(
        in_train_dataset,
        out_train_dataset,
        verbose=0)
    train_err = 1.0 - acc

    test_dataset = dst.Dataset()
    test_dataset.load_dataset(
        dataset_name=params.dataset_name,
        dataset_size=params.test_dataset_size,
        dataset_offset=params.test_dataset_offset,
        train_flag=False)

    in_test_dataset = test_dataset.in_dataset
    out_test_dataset = test_dataset.out_dataset

    loss, acc = model.evaluate(
        in_test_dataset,
        out_test_dataset,
        verbose=0)
    test_err = 1.0 - acc

    str_err = 'Trained model\n' \
              + '  Training error: ' \
              + str(round(train_err * 100, 2)) \
              + '%, Testing error: ' \
              + str(round(test_err * 100, 2)) + '%'

    print(str_err)
    utl.save_message(train_file, str_err + '\n', 'a')

    return


if __name__ == '__main__':
    prm.define_default_parameters(flags)
    app.run(main)
