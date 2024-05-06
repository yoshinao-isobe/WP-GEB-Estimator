#!/bin/bash
# 1:result_dir 2:net_arch 3:dataset 4:reg 5:dropout

# 2024/03/29, AIST
#  step 1 (train) : training simple classifiers for demonstrations
#  step 2 (measure) : measuring misclassification rates with random perturbations
#  step 3 (search) : searching for adversarial perturbations
#  step 4 (estimate) : estimating generalization error upper bounds

RESULT_DIR=$1
NET_ARCH=$2
MODEL_DIR=$2
DATASET_NAME=$3
REG_L2=$4
DROPOUT_RATE=$5

TRAIN_DATASET_SIZE=50000
TRAIN_DATASET_OFFSET=0
TEST_DATASET_SIZE=1000
TEST_DATASET_OFFSET=0

cd ../src || exit

# step 1
<<CCC
python train_main.py \
    --result_dir $RESULT_DIR \
    --dataset_name $DATASET_NAME \
    --net_arch_file net_arch/$NET_ARCH \
    --model_dir $MODEL_DIR \
    --train_dataset_size $TRAIN_DATASET_SIZE \
    --test_dataset_size $TEST_DATASET_SIZE \
    --train_dataset_offset $TRAIN_DATASET_OFFSET \
    --test_dataset_offset $TEST_DATASET_OFFSET \
    --epochs 50 --batch_size 100 \
    --regular_l2 $REG_L2 --dropout_rate $DROPOUT_RATE
CCC
# step 2

python measure_main.py \
    --result_dir $RESULT_DIR \
    --dataset_name $DATASET_NAME \
    --dataset_size $TEST_DATASET_SIZE \
    --dataset_offset $TEST_DATASET_OFFSET \
    --model_dir $MODEL_DIR \
    --perturb_sample_size 348 --batch_size 1000 \
    --perturb_ratios "0.001 0.002 0.003 0.005 0.007 0.01 0.02 0.03 0.05 0.07 0.1 0.2 0.3 0.5 0.7 1 2 3 5 7 10"\

# step 3

python search_main.py \
    --result_dir $RESULT_DIR \
    --batch_size 50 \
    --search_mode 1

# step 4

python estimate_main.py \
    --result_dir $RESULT_DIR \
    --delta 0.1 --delta0_ratio 0.5


cd ../examples || exit
