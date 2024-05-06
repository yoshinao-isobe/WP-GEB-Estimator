# 2024/03/29, AIST
# evaluating test-errors with random perturbations on wights
# (options for specifying parameters)

import dataset as dst


class OptParams(object):
    def __init__(self, f):
        self.random_seed = f.random_seed
        self.result_dir = f.result_dir
        self.measure_file = f.measure_file
        self.model_dir = f.model_dir

        # dataset
        self.dataset_name = f.dataset_name
        self.dataset_size = f.dataset_size
        self.dataset_offset = f.dataset_offset
        self.dataset_file = f.dataset_file
        self.dataset_fmt = f.dataset_fmt
        self.image_width = f.image_width
        self.image_height = f.image_height
        self.batch_size = f.batch_size

        # perturbation
        perturb_str_list = f.perturb_ratios.split()
        self.perturb_ratios = list(map(float, perturb_str_list))
        self.perturb_bn = f.perturb_bn

        # estimate
        self.perturb_sample_size = f.perturb_sample_size
        self.verbose_measure = f.verbose_measure

    def model_params(self):
        # information written with results in text files
        s = 'Parameters for random perturbations:\n'
        s = s + ' [global]\n'
        s = s + '  random_seed: ' + str(self.random_seed) + ','
        s = s + '  result_dir: ' + str(self.result_dir) + ',\n'
        s = s + '  model_dir: ' + str(self.model_dir) + ','
        s = s + '  measure_file: ' + str(self.measure_file) + '\n'

        s = s + ' [dataset]\n'
        s = s + '  dataset_name: ' + str(self.dataset_name) + ','
        s = s + '  dataset_size: ' + str(self.dataset_size) + ','
        s = s + '  dataset_offset: ' + str(self.dataset_offset) + ','
        s = s + '  batch_size: ' + str(self.batch_size) + ',\n'
        if self.dataset_name not in dst.Keras_dataset:
            s = s + '  dataset_file: ' + self.dataset_file + ',\n'
            s = s + '  dataset_fmt: ' + self.dataset_fmt + ','
            s = s + '  image_width: ' + str(self.image_width) + ','
            s = s + '  image_height: ' + str(self.image_height) + ',\n'

        s = s + ' [random perturbation]\n'
        s = s + '  batch_size: ' + str(self.batch_size) + ','
        s = s + '  perturb_sample_size: ' + str(self.perturb_sample_size) + '\n'
        return s


def define_default_parameters(f):

    model_dir_help \
        = 'specifies the directory for loading the pre-trained model or one of the following names:' \
          + '\'inception_v3\', \'inception_resnet_v2\', '\
          + '\'resnet50\', \'xception\', \'densenet121\','\
          + '\'densenet169\', \'densenet201\', \'vgg16\', \'vgg19\', '\
          + '\'nasnetlarge\', \'nasnetmobile\'}.'

    # global
    f.DEFINE_integer('random_seed', 1, 'specifies the random seed (the seed is not specified if 0).')
    f.DEFINE_string('result_dir', 'result', 'specifies the directory for saving the measurement results.')
    f.DEFINE_string('measure_file', 'measure', 'specifies the file name for saving measurement information.')
    f.DEFINE_string('model_dir', 'model', model_dir_help)

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
    f.DEFINE_integer(
        'batch_size', 0,
        'is the size of sub-datasets that the dataset is divided into (it is the dataset size if 0).')

    # measure
    f.DEFINE_string('perturb_ratios', '0.01 0.1 1', 'is the list of the ratios of perturbations (Delimiter　\' \').')
    f.DEFINE_integer('perturb_bn', 0, 'also perturbs the parameters (scale, shift) in batch normalization if 1.')
    f.DEFINE_integer('perturb_sample_size', 1215, 'is the size of random perturbation sample')
    f.DEFINE_integer('verbose_measure', 1, 'displays the progress in measuring test-errors with random perturbations if 1.')
