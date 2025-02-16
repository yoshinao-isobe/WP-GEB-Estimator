# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# train models (neural classifiers) for demonstrations
# (options for specifying parameters)


class InitParams(object):
    def __init__(self, f):
        self.random_seed = f.random_seed
        self.verbose = f.verbose
        self.result_dir = f.result_dir
        self.model_dir = f.model_dir
        self.train_file = f.train_file
        self.net_arch_file = f.net_arch_file

        # dataset
        self.dataset_name = f.dataset_name
        self.train_dataset_size = f.train_dataset_size
        self.test_dataset_size = f.test_dataset_size
        self.train_dataset_offset = f.train_dataset_offset
        self.test_dataset_offset = f.test_dataset_offset
        self.validation_ratio = f.validation_ratio

        # training initial models
        self.sigma = f.sigma
        self.batch_size = f.batch_size
        self.epochs = f.epochs
        self.dropout_rate = f.dropout_rate
        self.learning_rate = f.learning_rate
        self.decay_rate = f.decay_rate
        self.decay_steps = f.decay_steps
        self.early_stop = f.early_stop
        self.early_stop_delta = f.early_stop_delta
        self.early_stop_patience = f.early_stop_patience
        self.regular_l2 = f.regular_l2

    def model_params(self):
        # information written with results in text files
        s = 'Parameters for training:\n'
        s = s + ' [global]\n'
        s = s + '  random_seed: ' + str(self.random_seed) + ','
        s = s + '  result_dir: ' + str(self.result_dir) + ','
        s = s + '  train_file: ' + str(self.train_file) + '\n'
        s = s + ' [dataset]\n'
        s = s + '  dataset_name: ' + str(self.dataset_name) + ','
        s = s + '  train_dataset_size: ' + str(self.train_dataset_size) + ','
        s = s + '  train_dataset_offset: ' + str(self.train_dataset_offset) + ',\n'
        s = s + '  test_dataset_size: ' + str(self.test_dataset_size) + ','
        s = s + '  test_dataset_offset: ' + str(self.test_dataset_offset) + ',\n'
        s = s + '  validation_ratio: ' + str(self.validation_ratio) + '\n'
        s = s + ' [training]\n'
        s = s + '  net_arch_file: ' + str(self.net_arch_file) + ','
        s = s + '  model_dir: ' + str(self.model_dir) + '\n'
        s = s + '  sigma: ' + str(self.sigma) + ','
        s = s + '  batch_size: ' + str(self.batch_size) + ','
        s = s + '  epochs: ' + str(self.epochs) + ',\n'
        s = s + '  dropout_rate: ' + str(self.dropout_rate) + ','
        s = s + '  regular_l2: ' + str(self.regular_l2) + ',\n'
        s = s + '  learning_rate: ' + str(self.learning_rate) + ','
        s = s + '  decay_rate: ' + str(self.decay_rate) + ','
        s = s + '  decay_steps: ' + str(self.decay_steps) + ',\n'
        s = s + '  early_stop: ' + str(self.early_stop) + ','
        s = s + '  early_stop_delta: ' + str(self.early_stop_delta) + ','
        s = s + '  early_stop_patience: ' + str(self.early_stop_patience) + '\n'
        s = s + '\n'
        return s


# options
def define_default_parameters(f):

    # global
    f.DEFINE_integer('random_seed', 1, 'specifies the random seed (the seed is not specified if 0).')
    f.DEFINE_string('result_dir', 'result', 'specifies the directory for saving the results on training.')
    f.DEFINE_string('train_file', 'train', 'specifies the file name for saving training information.')
    f.DEFINE_string('net_arch_file', 'net_arch/cnn_s', 'specifies the csv-file name of network architecture.')
    f.DEFINE_string('model_dir', 'model', 'specifies the directory for saving the trained model.')
    f.DEFINE_integer('verbose', 1, 'displays the progress in training if 1.')

    # dataset
    f.DEFINE_string('dataset_name', 'mnist', 'specifies dataset name in {\'mnist\', \'fashion_mnist\', \'cifar10\'}.')
    f.DEFINE_integer('train_dataset_size', 50000, 'is the training dataset size.')
    f.DEFINE_integer('train_dataset_offset', 0, 'is the offset (start index) of the training dataset.')
    f.DEFINE_integer('test_dataset_size', 5000, 'is the testing dataset size.')
    f.DEFINE_integer('test_dataset_offset', 0, 'is the offset (start index) of the testing dataset.')
    f.DEFINE_float('validation_ratio', 0.1, 'specifies the ratio for validation in training dataset.')

    # training parameter
    f.DEFINE_float('sigma', 0.1, 'is the standard deviation of normal distribution used for initializing weights.')
    f.DEFINE_integer('batch_size', 100, 'specifies the batch size for training.')
    f.DEFINE_integer('epochs', 50, 'is the number of epochs for training.')
    f.DEFINE_float('dropout_rate', 0.0, 'is used as the dropout rate if it is not specified in \'net_arch_file\'.')
    f.DEFINE_float('regular_l2', 0.0, 'is used as l2-regularization if it is not specified in \'net_arch_file\'.')

    f.DEFINE_float('learning_rate', 0.01, 'specifies the (initial) learning rate.')
    f.DEFINE_float('decay_rate', 1.0, 'specifies the exponential decay rate of the learning rate.')
    f.DEFINE_integer('decay_steps', 0, 'specifies the steps of the 1st power of the decay_rate (no decay if 0).')
    f.DEFINE_integer('early_stop', 0, 'enables early-stopping if 1.')
    f.DEFINE_float('early_stop_delta', 0.0, 'specifies the minimum delta for early-stopping.')
    f.DEFINE_integer('early_stop_patience', 3, 'specifies the patience-epochs for early-stopping.')
