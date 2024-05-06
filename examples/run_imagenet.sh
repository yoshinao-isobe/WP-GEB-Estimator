#!/bin/bash
# 2024/03/29, AIST
#  step 1 (train) : training simple classifiers for demonstrations
#  step 2 (measure) : measuring misclassification rates with random perturbations
#  step 3 (search) : searching for adversarial perturbations
#  step 4 (estimate) : estimating generalization error upper bounds

# an example for estimating weight-perturbed generalization error bounds
# of pre-trained model "Inception-v3" by the dataset "ImageNet".
# (It is assumed that the validation dataset of ImageNet is saved
#  in the files "validation-*-of-00128" by the TFRecord format.)

cd ../src || exit

ImageNet_DIR="$HOME/datasets/imagenet/1k-tfrecords/validation"
python measure_main.py \
    --result_dir "result_imagenet" --dataset_name "imagenet" \
    --dataset_file "$ImageNet_DIR/validation-*-of-00128" \
    --dataset_size 1000 --batch_size 50 \
    --model_dir "inception_v3" \
    --perturb_ratios "0.0001" --perturb_sample_size 525

python search_main.py --result_dir "result_imagenet"

python estimate_main.py --result_dir "result_imagenet"

cd ../examples || exit
