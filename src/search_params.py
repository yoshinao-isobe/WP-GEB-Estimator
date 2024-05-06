# 2024/03/20, AIST
# searching for adversarial perturbations by wight-gradients
# (options for specifying parameters)


class OptParams(object):
    def __init__(self, f):
        self.random_seed = f.random_seed
        self.result_dir = f.result_dir
        self.search_file = f.search_file
        self.measure_file = f.measure_file
        self.batch_size = f.batch_size

        self.max_iteration = f.max_iteration
        self.search_mode = f.search_mode
        self.skip_search = f.skip_search
        self.verbose_search = f.verbose_search

    def model_params(self):
        # information written with results in text files
        s = 'Parameters for searching:\n'
        s = s + ' [global]\n'
        s = s + '  random_seed: ' + str(self.random_seed) + ','
        s = s + '  result_dir: ' + str(self.result_dir) + ',\n'
        s = s + '  measure_file: ' + str(self.measure_file) + ','
        s = s + '  search_file: ' + str(self.search_file) + '\n'

        s = s + ' [search perturbations]\n'
        s = s + '  batch_size: ' + str(self.batch_size) + ',\n'
        if self.skip_search == 0:
            s = s + '  search_mode: ' + str(self.search_mode) + ','
            s = s + '  max_iteration: ' + str(self.max_iteration) + ','
        else:
            s = s + '  skip_search: ' + str(self.skip_search) + '\n'
        return s


def define_default_parameters(f):

    # global
    f.DEFINE_integer('random_seed', 1, 'specifies the random seed (the seed is not specified if 0).')
    f.DEFINE_string('result_dir', 'result', 'specifies the directory for saving the search results.')
    f.DEFINE_string('measure_file', 'measure', 'specifies the file name for loading random perturbation information.')
    f.DEFINE_string('search_file', 'search', 'specifies the file name for saving search results.')

    f.DEFINE_integer(
        'batch_size', 10,
        'is the size of sub-datasets that the dataset is divided into (it is the dataset size if 0).')

    # search
    f.DEFINE_integer('skip_search', 0, 'skips the search for adversarial perturbations if 1.')
    f.DEFINE_integer('search_mode', 0,
                     'specifies the search-mode in {0,1} (0:FGSM, 1:Iterative-FGSM)')
    f.DEFINE_integer('max_iteration', 20, 'specifies the maximum iteration for searching adversarial perturbations.')
    f.DEFINE_integer('verbose_search', 1, 'displays the progress in searching adversarial perturbations if 1.')
