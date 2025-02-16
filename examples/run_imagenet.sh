#!/bin/bash

# Copyright (C) 2025
# National Institute of Advanced Industrial Science and Technology (AIST)

#  step 1 (train) : skip (a pre-trained model is used.)
#  step 2 (search) : searching for adversarial perturbations
#  step 3 (measure) : measuring misclassification rates with random perturbations
#  step 4 (estimate) : estimating generalization error upper bounds

# an example for estimating weight-perturbed generalization error bounds
# of pre-trained model "Inception-v3" by the dataset "ImageNet".
# (It is assumed that the validation dataset of ImageNet is saved
#  in the files "validation-*-of-00128" by the TFRecord format.)

cd ../src || exit

# Previously save ImageNet data in ImageNet_DIR by TF-Record format.
ImageNet_DIR="$HOME/datasets/imagenet/1k-tfrecords/validation"

# with search of adversarial perturbations by FGSM
python search_main.py \
    --skip_search 0 \
    --result_dir "result_imagenet" \
    --dataset_name "imagenet" \
    --dataset_file "$ImageNet_DIR/validation-*-of-00128" \
    --model_dir "inception_v3" \
    --dataset_size 1000 \
    --perturb_ratios "0.0001"

python measure_main.py \
    --result_dir "result_imagenet" \
    --err_thr 0.02

python estimate_main.py --result_dir "result_imagenet"

# without search
python search_main.py \
    --skip_search 1 \
    --result_dir "result_imagenet" \
    --dataset_name "imagenet" \
    --dataset_file "$ImageNet_DIR/validation-*-of-00128" \
    --model_dir "inception_v3" \
    --dataset_size 1000 \
    --perturb_ratios "0.0001"

python measure_main.py \
    --result_dir "result_imagenet" \
    --err_thr 0.02

python estimate_main.py --result_dir "result_imagenet"

cd ../examples || exit
