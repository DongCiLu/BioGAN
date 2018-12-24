# Usage:
# ./launch_jobs.sh ${run_mode} ${version} ${NUM_STEPS} ${gpu_unit}
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

base_dir="../bio_datasets/supervised_128"
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
	--batch_size=128 \
	--learning_rate=1e-4 \
        --alsologtostderr
fi
Banner "Finished ${run_mode} for ${NUM_EPOCHS} steps."
