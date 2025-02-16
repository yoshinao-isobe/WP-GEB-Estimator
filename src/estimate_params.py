# Copyright (C) 2024
# National Institute of Advanced Industrial Science and Technology (AIST)

# evaluating generalization error upper bounds with perturbations on weights
# (options for specifying parameters)


class OptParams(object):
    def __init__(self, f):
        self.result_dir = f.result_dir
        self.measure_file = f.measure_file
        self.estimate_file = f.estimate_file

        # evaluation
        self.max_nm = f.max_nm
        self.eps_nm = f.eps_nm

    def model_params(self):
        # information written with results in text files
        s = 'Parameters for estimating generalization bounds:\n'
        s = s + ' [global]\n'
        s = s + '  result_dir: ' + str(self.result_dir) + ',\n'
        s = s + '  measure_file: ' + str(self.measure_file) + ','
        s = s + '  estimate_file: ' + str(self.estimate_file) + '\n'

        s = s + ' [evaluation]\n'
        s = s + '  max_nm: ' + str(self.max_nm) + ','
        s = s + '  eps_nm: ' + str(self.eps_nm) + '\n'
        s = s + '\n'
        return s


def define_default_parameters(f):

    # global
    f.DEFINE_string('result_dir', 'result', 'specifies the directory for saving the estimate results.')
    f.DEFINE_string('measure_file', 'measure', 'specifies the file name for loading results on perturbations.')
    f.DEFINE_string('estimate_file', 'estimate', 'specifies the file name for saving generalization bounds.')

    # evaluation parameter
    f.DEFINE_integer('max_nm', 10, "is the max iterations in the Newton's method for inv_kl.")
    f.DEFINE_float('eps_nm', 0.0001, "is the threshold in the Newton's method for inv_kl.")
