# 2024/03/29, AIST
# evaluating generalization error upper bounds with perturbations on wights
# (options for specifying parameters)


class OptParams(object):
    def __init__(self, f):
        self.result_dir = f.result_dir
        self.search_file = f.search_file
        self.estimate_file = f.estimate_file

        # evaluation
        self.delta = f.delta
        self.delta0_ratio = f.delta0_ratio
        self.max_nm = f.max_nm
        self.eps_nm = f.eps_nm

    def model_params(self):
        # information written with results in text files
        s = 'Parameters for estimating generalization error bounds:\n'
        s = s + ' [global]\n'
        s = s + '  result_dir: ' + str(self.result_dir) + ',\n'
        s = s + '  search_file: ' + str(self.search_file) + ','
        s = s + '  estimate_file: ' + str(self.estimate_file) + '\n'

        s = s + ' [evaluation]\n'
        s = s + '  delta: ' + str(self.delta) + ','
        s = s + '  delta0_ratio: ' + str(self.delta0_ratio) + ',\n'
        s = s + '  max_nm: ' + str(self.max_nm) + ','
        s = s + '  eps_nm: ' + str(self.eps_nm) + '\n'
        s = s + '\n'
        return s


def define_default_parameters(f):

    # global
    f.DEFINE_string('result_dir', 'result', 'specifies the directory for saving the estimate results.')
    f.DEFINE_string('search_file', 'search', 'specifies the file name for loading results on perturbations.')
    f.DEFINE_string('estimate_file', 'estimate', 'specifies the file name for saving generalization bounds.')

    # evaluation parameter
    f.DEFINE_float('delta', 0.1,
                   'is the probability that the perturbed generalization error is higher than the upper bound.'
                   + ' (i.e. (1 - delta) is the confidence level.)')
    f.DEFINE_float('delta0_ratio', 0.5,
                   'is the ratio in \'delta\' that the perturbed testing error is higher than the upper bound.'
                   + ' (i.e. (1 - delta * delta0_ratio) is the confidence level.)')
    f.DEFINE_integer('max_nm', 10, "is the max iterations in the Newton's method for inv_kl.")
    f.DEFINE_float('eps_nm', 0.0001, "is the threshold in the Newton's method for inv_kl.")
