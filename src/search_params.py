# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# searching for adversarial perturbations by weight-gradients
# (options for specifying parameters)

import dataset as dst


class OptParams(object):
    def __init__(self, f):

        # dataset
        self.dataset_name = f.dataset_name
        self.dataset_size = f.dataset_size
        self.dataset_offset = f.dataset_offset
        self.dataset_file = f.dataset_file
        self.dataset_fmt = f.dataset_fmt
        self.image_width = f.image_width
        self.image_height = f.image_height

        # global
        self.random_seed = f.random_seed
        self.result_dir = f.result_dir
        self.search_file = f.search_file
        self.model_dir = f.model_dir

        # perturbation
        perturb_str_list = f.perturb_ratios.split()
        self.perturb_ratios = list(map(float, perturb_str_list))
        self.perturb_bn = f.perturb_bn

        self.batch_size = f.batch_size
        self.max_iteration = f.max_iteration
        self.search_mode = f.search_mode
        self.skip_search = f.skip_search
        self.verbose_search = f.verbose_search

    def model_params(self):
        # information written with results in text files
        s = 'Parameters for searching:\n'
        s = s + ' [dataset]\n'
        s = s + '  dataset_name: ' + str(self.dataset_name) + ','
        s = s + '  dataset_size: ' + str(self.dataset_size) + ','
        s = s + '  dataset_offset: ' + str(self.dataset_offset) + ',\n'
        # s = s + '  batch_size: ' + str(self.batch_size) + ',\n' koko

        if self.dataset_name not in dst.Keras_dataset:
            s = s + '  dataset_file: ' + self.dataset_file + ',\n'
            s = s + '  dataset_fmt: ' + self.dataset_fmt + ','
            s = s + '  image_width: ' + str(self.image_width) + ','
            s = s + '  image_height: ' + str(self.image_height) + ',\n'

        s = s + ' [global]\n'
        s = s + '  random_seed: ' + str(self.random_seed) + ','
        s = s + '  result_dir: ' + str(self.result_dir) + ',\n'
        # s = s + '  measure_file: ' + str(self.measure_file) + ',' koko
        s = s + '  search_file: ' + str(self.search_file) + '\n'

        s = s + ' [search perturbations]\n'
        s = s + '  batch_size: ' + str(self.batch_size) + ',\n'
        s = s + '  perturb_ratios: ' + str(self.perturb_ratios) + ',\n'
        if self.skip_search == 0:
            s = s + '  search_mode: ' + str(self.search_mode) + ','
            s = s + '  max_iteration: ' + str(self.max_iteration) + ','
        else:
            s = s + '  skip_search: ' + str(self.skip_search) + '\n'
        return s


def define_default_parameters(f):

    # dataset
    f.DEFINE_string(
        'dataset_name', 'mnist', 'specifies dataset name in {\'mnist\', \'fashion_mnist\', \'cifar10\', \'imagenet\'}.')
    f.DEFINE_string(
        'dataset_file', '~/imagenet/1k-tfrecords/validation-*-of-00128',
        'specifies files for loading the dataset saved in the tfrecord format.')
    f.DEFINE_string(
        'dataset_fmt', 'tfrecord', 'specifies the format of dataset file in {\'tfrecord\'}.')
    f.DEFINE_integer('image_width', 0, 'is the width of original images (0 for images given by Keras.')
    f.DEFINE_integer('image_height', 0, 'is the height of original images (0 for images given by Keras.')
    f.DEFINE_integer('dataset_size', 5000, 'is the dataset size for searching adversarial perturbations.')
    f.DEFINE_integer('dataset_offset', 0, 'is the offset (start index) of the dataset.')

    model_dir_help \
        = 'specifies the directory for loading the pre-trained model or one of the following names:' \
          + '\'inception_v3\', \'inception_resnet_v2\', '\
          + '\'resnet50\', \'xception\', \'densenet121\','\
          + '\'densenet169\', \'densenet201\', \'vgg16\', \'vgg19\', '\
          + '\'nasnetlarge\', \'nasnetmobile\'}.'

    # global
    f.DEFINE_integer('random_seed', 1, 'specifies the random seed (the seed is not specified if 0).')
    f.DEFINE_string('result_dir', 'result', 'specifies the directory for saving the search results.')
    f.DEFINE_string('model_dir', 'model', model_dir_help)
    f.DEFINE_string('search_file', 'search', 'specifies the file name for saving search results.')

    f.DEFINE_integer(
        'batch_size', 10,
        'is the size of sub-datasets that the dataset is divided into (it is the dataset size if 0).')

    # search
    f.DEFINE_string('perturb_ratios', '0.01 0.1 1', 'is the list of the ratios of perturbations (Delimiter　\' \').')
    f.DEFINE_integer('perturb_bn', 0, 'also perturbs the parameters (scale, shift) in batch normalization if 1.')

    f.DEFINE_integer('skip_search', 0, 'skips the search for adversarial perturbations if 1.')
    f.DEFINE_integer('search_mode', 0,
                     'specifies the search-mode in {0,1} (0:FGSM, 1:Iterative-FGSM)')
    f.DEFINE_integer('max_iteration', 20, 'specifies the maximum iteration for searching adversarial perturbations.')
    f.DEFINE_integer('verbose_search', 1, 'displays the progress in searching adversarial perturbations if 1.')
