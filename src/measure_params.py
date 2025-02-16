# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

# evaluating test-errors with random perturbations on weights
# (options for specifying parameters)


class OptParams(object):
    def __init__(self, f):
        self.random_seed = f.random_seed
        self.result_dir = f.result_dir
        self.measure_file = f.measure_file
        self.search_file = f.search_file

        # estimate
        self.batch_size = f.batch_size
        self.err_thr = f.err_thr
        self.perturb_sample_size = f.perturb_sample_size
        self.verbose_measure = f.verbose_measure

        # unconfidence
        self.delta = f.delta
        self.delta0_ratio = f.delta0_ratio

    def model_params(self):
        # information written with results in text files
        s = 'Parameters for random perturbations:\n'
        s = s + ' [global]\n'
        s = s + '  random_seed: ' + str(self.random_seed) + ','
        s = s + '  result_dir: ' + str(self.result_dir) + ',\n'
        s = s + '  search_file: ' + str(self.search_file) + ', '
        s = s + '  measure_file: ' + str(self.measure_file) + '\n'

        s = s + ' [random perturbation]\n'
        s = s + '  batch_size: ' + str(self.batch_size) + ','
        s = s + '  err_thr: ' + str(self.err_thr) + ','
        s = s + '  perturb_sample_size: ' + str(self.perturb_sample_size) + '\n'

        s = s + ' [unconfidence]\n'
        s = s + '  delta: ' + str(self.delta) + ','
        s = s + '  delta0_ratio: ' + str(self.delta0_ratio) + ',\n'
        s = s + '\n'
        return s


def define_default_parameters(f):
    f.DEFINE_integer('random_seed', 1, 'specifies the random seed (the seed is not specified if 0).')
    f.DEFINE_string('result_dir', 'result', 'specifies the directory for saving the search results.')

    f.DEFINE_integer(
        'batch_size', 0,
        'is the size of sub-datasets that the dataset is divided into (it is the dataset size if 0).')

    f.DEFINE_string('search_file', 'search', 'specifies the file name for loading search results.')
    f.DEFINE_string('measure_file', 'measure', 'specifies the file name for saving measure results.')

    # measure
    f.DEFINE_integer('perturb_sample_size', 0,
                     'is the size of random perturbation sample (if 0 then the size is decided from err_thr)')
    f.DEFINE_integer('verbose_measure', 1,
                     'displays the progress in measuring test-errors with random perturbations if 1.')
    f.DEFINE_float('err_thr', 0.01, 'is the acceptable threshold of error-rate')

    # evaluation parameter
    f.DEFINE_float('delta', 0.1,
                   'is the probability that the perturbed generalization error is higher than the upper bound.'
                   + ' (i.e. (1 - delta) is the confidence level.)')
    f.DEFINE_float('delta0_ratio', 0.5,
                   'is the ratio in \'delta\' that the perturbed testing error is higher than the upper bound.'
                   + ' (i.e. (1 - delta * delta0_ratio) is the confidence level.)')
