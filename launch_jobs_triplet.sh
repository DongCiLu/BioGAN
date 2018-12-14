# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#!/bin/bash
#
# This script performs the following operations:
# 1. Downloads the MNIST dataset.
# 2. Trains an unconditional, conditional, or InfoGAN model on the MNIST
#    training set.
# 3. Evaluates the models and writes sample images to disk.
#
# These examples are intended to be fast. For better final results, tune
# hyperparameters or train longer.
#
# NOTE: Each training step takes about 0.5 second with a batch size of 32 on
# CPU. On GPU, it takes ~5 milliseconds.
#
# With the default batch size and number of steps, train times are:
#
#   unconditional: CPU: 800  steps, ~10 min   GPU: 800  steps, ~1 min
#   conditional:   CPU: 2000 steps, ~20 min   GPU: 2000 steps, ~2 min
#   infogan:       CPU: 3000 steps, ~20 min   GPU: 3000 steps, ~6 min
#
# Usage:
# ./launch_jobs.sh ${run_mode} ${version}
set -e

# define run mode
run_mode=$1
if ! [[ "$run_mode" =~ ^(test|triplet_training) ]]; then
    echo "'run_mode' must be one of: 'test', 'triplet_training'."
    exit
fi

# define version for the log files
version=$2
if [[ "$version" == "" ]]; then
    version="test"
fi

# define number of steps to run the experiment
NUM_STEPS=$3
if [[ "$NUM_STEPS" == "" ]]; then
    NUM_STEPS=100
fi

# define which GPU to run on
gpu_unit=$4
if [[ "$gpu_unit" == "" ]]; then
    echo "use default gpu (GPU0)."
    gpu_unit=0
fi

export CUDA_VISIBLE_DEVICES=$gpu_unit
PYTHON="python"

# Location of the git repository.
git_repo="../Tensorflow-models"

# Location of the src directory
src_dir="."

# Base name for where the checkpoint and logs will be saved to.
TRAIN_DIR=celegans-models/${version}

# Base name for where the evaluation images will be saved to.
EVAL_DIR=celegans-models/eval/${version}

# Where the dataset is saved to.
DATASET_DIR=celegans-128-supervised

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/slim

# A helper function for printing pretty output.
Banner () {
    local text=$1
    local green='\033[0;32m'
    local nc='\033[0m'  # No color.
    echo -e "${green}${text}${nc}"
}

Banner "Starting ${run_mode} for ${NUM_EPOCHS} steps..."

# Run temporary tests.
if [[ "$run_mode" == "test" ]]; then
    ${PYTHON} "${src_dir}/data_provider_test.py" 
fi

DATA_CONFIG=128_1to3_1.0_1

base_dir="../bio_datasets/supervised_128/triplet"
exp_train_size=$( find "${base_dir}/train" -type f | wc -l )
exp_test_size=$( find "${base_dir}/test" -type f | wc -l )

# Run training.
if [[ "$run_mode" == "triplet_training" ]]; then
    ${PYTHON} "${src_dir}/classification/train.py" \
        --train_log_dir=${TRAIN_DIR} \
        --dataset_dir=${DATASET_DIR} \
        --data_config="${DATA_CONFIG}_${exp_train_size}_${exp_test_size}" \
        --hyper_mode="regular" \
        --mode="triplet_train" \
        --network="dconvnet" \
        --max_number_of_steps=${NUM_STEPS} \
        --alsologtostderr
fi
Banner "Finished ${run_mode} for ${NUM_EPOCHS} steps."
