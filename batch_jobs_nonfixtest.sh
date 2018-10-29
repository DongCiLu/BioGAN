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
# cd models/research/gan/mnist
# ./launch_jobs.sh ${gan_type} ${git_repo}
set -e


### IMPORTANT: network of same size can only have 1 running instance at a time #
network_size=$1
if ! [[ "$network_size" =~ ^(128|32) ]]; then
    echo "'network_size' must be one of: 128 or 32."
    exit
fi

# Location of the git repository.
git_repo="/data/Tensorflow-models"

gpu_unit=$2
if [[ "$gpu_unit" == "" ]]; then
    echo "use default gpu (GPU1)."
    gpu_unit=1
fi

export CUDA_VISIBLE_DEVICES=$gpu_unit

# Base name for where the checkpoint and logs will be saved to.
TRAIN_DIR=celegans-model

# Base name for where the evaluation images will be saved to.
EVAL_DIR=celegans-model/eval

export PYTHONPATH=$PYTHONPATH:$git_repo:$git_repo/research:$git_repo/research/slim

# A helper function for printing pretty output.
Banner () {
    local text=$1
    local green='\033[0;32m'
    local nc='\033[0m'  # No color.
    echo -e "${green}${text}${nc}"
}

# parameter list for batch mode
CLASSIFICATION_DATASET_DIR="celegans-${network_size}-supervised"
NUM_STEPS=50000
training_mode=("raw" "trans") # without or with transfer learning
# dataset_ratio=("1to1" "1to3") # rosette to non-rosette ratio
dataset_ratio=("1to3") # set to 1:3 for now
train_ratio=(1.0 0.9 0.8 0.7 0.6 0.5 0.4 0.3 0.2 0.1) # how many training data we will use to train the network, in percentage.
exp_id=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20) # 5 experiments for each parameter settings

if [[ "${network_size}" == 128 ]]; then
    HYPER_MODE="regular"
else
    HYPER_MODE="tiny"
fi

for i in "${training_mode[@]}"
do
    if [[ "${i}" == "raw" ]]; then
        WARM_START_PARA=0
    else
        WARM_START_PARA=1
    fi
    for j in "${dataset_ratio[@]}"
    do
        for k in "${train_ratio[@]}"
        do
            for l in "${exp_id[@]}"
            do
                ID="${i}_${j}_${k}_exp${l}"
                CLASSIFICATION_TRAIN_DIR="${TRAIN_DIR}/${network_size}-${ID}"
                CLASSIFICATION_EVAL_DIR="${EVAL_DIR}/${network_size}-${ID}"
                DATA_CONFIG="${network_size}_${j}_${k}_1" # dataset number is 1

                # Prepare data.
                Banner "Preparing data..."
                base_dir="datasets/classification/supervised_${network_size}"
                rm -rf "${base_dir}/train/"
                rm -rf "${base_dir}/test/"
                rm -rf "${base_dir}/tfrecord/"
                python "seperate_data_nonfixtest.py" ${network_size} ${j} ${k}
                exp_train_size=$( find "${base_dir}/train" -type f | wc -l )
                exp_test_size=$( find "${base_dir}/test" -type f | wc -l )
                mkdir "${base_dir}/tfrecord/"
                python "build_image_data.py" \
                    --train_directory "${base_dir}/train" \
                    --validation_directory "${base_dir}/test" \
                    --output_directory "${base_dir}/tfrecord" \
                    --labels_file "${base_dir}/celegans_label.txt"
                mv "${base_dir}/tfrecord/train-00000-of-00001" \
                    "${base_dir}/tfrecord/celegans-train_${j}_${k}_1.tfrecord"
                mv "${base_dir}/tfrecord/validation-00000-of-00001" \
                    "${base_dir}/tfrecord/celegans-test_${j}_${k}_1.tfrecord"

                # Run training.
                Banner "Starting training classifier for ${NUM_STEPS} steps..."
                python "classification/train.py" \
                    --train_log_dir=${CLASSIFICATION_TRAIN_DIR} \
                    --dataset_dir=${CLASSIFICATION_DATASET_DIR} \
                    --data_config=${DATA_CONFIG}_${exp_train_size}_${exp_test_size} \
                    --hyper_mode=${HYPER_MODE} \
                    --network="dconvnet" \
                    --max_number_of_steps=${NUM_STEPS} \
                    --warm_start=${WARM_START_PARA}
                Banner "Finished training classifier ${ID} for ${NUM_STEPS} steps."

                # Clean up.
                Banner "Cleaning up..."
                rm -rf "${base_dir}/train/"
                rm -rf "${base_dir}/test/"
                rm -rf "${base_dir}/tfrecord/"
            done
        done
    done
done
