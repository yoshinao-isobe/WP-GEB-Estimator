#!/bin/bash

# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

#  step 1 (train) : training simple classifiers for demonstrations
#  step 3 (search) : searching for adversarial perturbations
#  step 2 (measure) : measuring misclassification rates with random perturbations
#  step 4 (estimate) : estimating generalization error upper bounds

# a simple example for training a multilayer perceptron with batch-normalization
# and estimating weight-perturbed generalization error bounds.


cd ../src || exit

python train_main.py  --net_arch_file "net_arch/mlp_s_bn"

# with search of adversarial perturbations by FGSM
python search_main.py  --skip_search 0
python measure_main.py
python estimate_main.py

# without search
python search_main.py  --skip_search 1
python measure_main.py
python estimate_main.py

cd ../examples || exit

