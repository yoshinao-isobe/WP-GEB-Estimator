#!/bin/bash
# 2024/03/29, AIST
#  step 1 (train) : training simple classifiers for demonstrations
#  step 2 (measure) : measuring misclassification rates with random perturbations
#  step 3 (search) : searching for adversarial perturbations
#  step 4 (estimate) : estimating generalization error upper bounds

# a simple example for training a multilayer perceptron with batch-normalization
# and estimating weight-perturbed generalization error bounds.


cd ../src || exit

python train_main.py  --net_arch_file "net_arch/mlp_s_bn"
python measure_main.py
python search_main.py
python estimate_main.py

cd ../examples || exit

