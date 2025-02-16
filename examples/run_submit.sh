#!/bin/bash
# 1:dataset_name 2:net_arch
# 2025 AIST

DIR_SUB_RESULT=result_$1_$2

DIR_PWD=$(pwd)
DIR_RESULT=$DIR_PWD/../results/$DIR_SUB_RESULT

if [ ! -d $DIR_RESULT ]; then
  mkdir $DIR_RESULT
fi

./run_option.sh $DIR_SUB_RESULT/result1 $2 $1 0.0 0.0
./run_option.sh $DIR_SUB_RESULT/result2 $2 $1 0.001 0.0
# ./run.sh $DIR_SUB_RESULT/result3 $2 $1 0.0 0.1
