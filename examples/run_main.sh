#!/bin/bash
# run the scripts by specifying the dataset and the model

./run_submit.sh mnist cnn_s

<<CCC
./run_submit.sh fashion_mnist cnn_s
./run_submit.sh cifar10 mlp_m
./run_submit.sh cifar10 cnn_m

./run_submit.sh imagenet densenet121
./run_submit.sh imagenet densenet169
./run_submit.sh imagenet densenet201
./run_submit.sh imagenet inception_resnet_v2
./run_submit.sh imagenet inception_v3
./run_submit.sh imagenet nasnetlarge
./run_submit.sh imagenet nasnetmobile
./run_submit.sh imagenet resnet50
./run_submit.sh imagenet vgg16
./run_submit.sh imagenet vgg19
./run_submit.sh imagenet xception
CCC