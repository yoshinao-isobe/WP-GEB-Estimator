# 2024/03/29, AIST
# construct the neural network specified by net_arch_dict_list

import tensorflow as tf
import utils as utl


# =================================================
# construction of neural networks
# =================================================

def net_const(
        in_shape,
        # out_size,
        net_arch_dict_list,
        sigma=0.1,
        regular_l2=0.0,
        dropout_rate=0.0):

    # initializing parameters by Gaussian distribution
    # weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=None)
    # bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=None)

    inputs = tf.keras.layers.Input(shape=in_shape)

    for i in range(len(net_arch_dict_list)):
        lyr = net_arch_dict_list[i]

        if lyr['regular_l2'] != '':
            lyr_regular_l2 = float(lyr['regular_l2'])
        else:
            lyr_regular_l2 = regular_l2

        if lyr['int_tuple'] != '':
            int_tuple = utl.str_to_int_tuple(lyr['int_tuple'])
        else:
            int_tuple = (4, 4)

        # --- Flatten ---
        if lyr['type'] == 'Flatten':
            layer = tf.keras.layers.Flatten()

        # --- Dense ---
        elif lyr['type'] == 'Dense':
            units = int(lyr['units'])

            '''
            if i == len(net_arch_dict_list) - 1 and units != out_size:
                units = out_size
                print('The number of units in the final dense layer: ', units)
            '''

            weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=None)
            bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=None)

            layer = tf.keras.layers.Dense(
                units,
                kernel_regularizer=tf.keras.regularizers.l2(lyr_regular_l2),
                activation=lyr['activation'],
                kernel_initializer=weight_init,
                bias_initializer=bias_init)

        # --- Dense no bias ---
        elif lyr['type'] == 'DenseNoBias':
            units = int(lyr['units'])

            '''
            if i == len(net_arch_dict_list) - 1 and units != out_size:
                units = out_size
                print('The number of units in the final dense layer: ', units)
            '''

            weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=None)

            layer = tf.keras.layers.Dense(
                units,
                use_bias=False,
                kernel_regularizer=tf.keras.regularizers.l2(lyr_regular_l2),
                activation=lyr['activation'],
                kernel_initializer=weight_init)

        # --- Conv2D ---
        elif lyr['type'] == 'Conv2D':

            weight_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=None)
            bias_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=sigma, seed=None)

            layer = tf.keras.layers.Conv2D(
                int(lyr['filters']),
                int_tuple,
                kernel_regularizer=tf.keras.regularizers.l2(lyr_regular_l2),
                activation=lyr['activation'],
                # padding='same',
                kernel_initializer=weight_init,
                bias_initializer=bias_init)

        # --- MaxPooling2D ---
        elif lyr['type'] == 'MaxPooling2D':

            layer = tf.keras.layers.MaxPooling2D(
                pool_size=int_tuple)

        # --- Dropout ---
        elif lyr['type'] == 'Dropout':

            if lyr['rate'] != '':
                rate = float(lyr['rate'])
            else:
                rate = dropout_rate

            layer = tf.keras.layers.Dropout(rate)

        # --- BatchNormalization ---
        elif lyr['type'] == 'BatchNormalization':

            layer = tf.keras.layers.BatchNormalization()

        # --- Activation ---
        elif lyr['type'] == 'Activation':

            layer = tf.keras.layers.Activation(activation=lyr['activation'])

        # --- etc ---
        else:
            print('Error: the layer type ({}) is not supported.'.format(lyr['type']))
            layer = None
            exit(1)

        if i == 0:
            x = layer(inputs)
        else:
            x = layer(x)

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model
