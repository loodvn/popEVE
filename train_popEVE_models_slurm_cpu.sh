#!/bin/bash 
#SBATCH --cpus-per-task=4

# For typical short jobs
#SBATCH -t 0-11:59 # time (D-HH:MM)
#SBATCH --mem=40G
#SBATCH -p short

##SBATCH --mail-type=TIME_LIMIT_80,TIME_LIMIT,FAIL,ARRAY_TASKS

#SBATCH --output=./logs/slurm/%x/%x-%A_%3a-%u.out  # Nice tip: using %3a to pad to 3 characters (23 -> 023) so that filenames sort properly
#SBATCH --error=./logs/slurm/%x/%x-%A_%3a-%u.err

#SBATCH --job-name="popeve_slurmcpu_1b_chunk100"
#SBATCH --array=0-65 # Take CSV line numbers minus 2 (they're 1-based and ignore header row)
# TODO ESM outstanding: 479-6912, and these: 444,446,450,457,461,473,386,400,406,284,339,36
# Fail on first error
set -e

# Example
# export mapping_file="example_mapping.csv"
# export results_dir="./results/example_mapping/"

# 1a TODO

# 1b
export mapping_file="mapping_benchmarking_ESM1V_20240427.csv"
export results_dir="./results/mapping_benchmarking_ESM1V_20240427/"

# 2a
# export mapping_file='mapping_snv_EVE_benchmarking_20240427.csv'
# export results_dir="./results/mapping_snv_EVE_benchmarking_20240427/"

# 2b:
# export mapping_file="mapping_snv_ESM1V_benchmarking_20240427.csv"
# export results_dir="./results/mapping_snv_ESM1V_benchmarking_20240427/"

# 3a:
# export mapping_file="mapping_snv_gf_EVE_benchmarking_20240427.csv"
# export results_dir="./results/mapping_snv_gf_EVE_benchmarking_20240427/"

export losses_and_lengthscales_directory="${results_dir}/losses_and_lengthscales/"
export scores_directory="$results_dir/scores/"
export states_directory="$results_dir/states/"

# Check if directories exist and create them if not
mkdir -p "$losses_and_lengthscales_directory"
mkdir -p "$scores_directory"
mkdir -p "$states_directory"

# One index at a time:
# /n/groups/marks/users/lood/mambaforge/envs/popeve_env/bin/python train_popEVE.py \
#     --mapping_file "$mapping_file" \
#     --gene_index "$SLURM_ARRAY_TASK_ID" \
#     --losses_dir "$losses_and_lengthscales_directory" \
#     --scores_dir "$scores_directory" \
#     --model_states_dir "$states_directory" \
#     --debug \
#     --skip_slow 12  # Skip any ones running longer than 12 hours

# export batch_size=100
# export max_index=7000
# # Loop through batch of indices (could also parallelise this via python)
# # Go from batch_index * batch_size to (batch_index + 1) * batch_size
# for ((i = 0 ; i < $batch_size ; i++)); do
#     index=$(($SLURM_ARRAY_TASK_ID * $batch_size + $i))
#     # Remember to break if we've hit the max
#     if [ $index -ge $max_index ]; then
#         break
#     fi
#     /n/groups/marks/users/lood/mambaforge/envs/popeve_env/bin/python train_popEVE.py \
#         --mapping_file "$mapping_file" \
#         --gene_index "$SLURM_ARRAY_TASK_ID" \
#         --losses_dir "$losses_and_lengthscales_directory" \
#         --scores_dir "$scores_directory" \
#         --model_states_dir "$states_directory" \
#         --debug \
#         --skip_slow 4  # Skip any ones running longer than 12 hours
# done

# Made num_cpus 6 instead of 4 hoping that there's some IO overhead and we win extra speed
# Lood: Actually taking this back down to 4 to be conservative (we had problems before with CPU oversubscription)
/n/groups/marks/users/lood/mambaforge/envs/popeve_env/bin/python train_popeve_o2.py \
        --chunk_index "$SLURM_ARRAY_TASK_ID" \
        --script_chunk_size 100 \
        --num_cpus 4 \
        --mapping_file "$mapping_file" \
        --losses_dir "$losses_and_lengthscales_directory" \
        --scores_dir "$scores_directory" \
        --model_states_dir "$states_directory" \
        --debug \
        --skip_slow 4  # Skip any ones running longer than 4 hours 
        # (if there are many of these, will have to run them over GPUs because we won't get through them on O2)