#!/usr/bin/env python
# Copied from Aaron's get_alignment_stats.py in his EVE repo
import os
import subprocess
import sys
from functools import partial
from contextlib import contextmanager,redirect_stderr,redirect_stdout

import pandas as pd
from tqdm.auto import tqdm
from multiprocessing import Pool

sys.path.append("/home/ubuntu/popEVE/")
from train_popEVE import train

@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, 'w') as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)

def s3_cp_file(from_path, to_path, silent=False):
    max_runs = 3
    run = 0
    error = Exception()
    while run < max_runs:
        try:
            return subprocess.run(
                ["aws", "s3", "cp", from_path, to_path],
                check=True,
                stdout=subprocess.DEVNULL if silent else None,
                stderr=subprocess.DEVNULL if silent else None,
            )
        except subprocess.CalledProcessError as e:
            error = e
            continue
        else:
            break
        finally:
            run += 1
    else:
        raise error

def s3_sync_folder(from_path, to_path, silent=False):
    max_runs = 3
    run = 0
    error = Exception()
    while run < max_runs:
        try:
            return subprocess.run(
                ["aws", "s3", "sync", from_path, to_path],
                check=True,
                stdout=subprocess.DEVNULL if silent else None,
                stderr=subprocess.DEVNULL if silent else None,
            )
        except subprocess.CalledProcessError as e:
            error = e
            continue
        else:
            break
        finally:
            run += 1
    else:
        raise error

tier = 1
silent = False  # Debugging

if tier == 0:
    # Testing
    mapping_file_name = "example_mapping.csv"
    mapping_df = pd.read_csv(mapping_file_name)
    run_name = "example_mapping"

# popEVE rerun with various small changes
elif tier == 1:
    # s3_cp_file("s3://markslab-us-east-2/colabfold/output/popeve/missing_genes_with_priority.tsv", ".")
    
    # Read mapping file (just kept this on git for simplicity)
    mapping_file_name = "mapping_benchmarking_EVE_20240427.csv"
    mapping_df = pd.read_csv(mapping_file_name)
    run_name = "mapping_benchmarking_EVE_20240427"

results_directory = f"/tmp/results/{run_name}/"
os.makedirs(results_directory, exist_ok=True)

losses_and_scales_directory = f"{results_directory}/losses_and_lengthscales/"
scores_directory = f"{results_directory}/scores/"
model_states_directory = f"{results_directory}/model_states/"
for dir in [losses_and_scales_directory, scores_directory, model_states_directory]:
    os.makedirs(dir, exist_ok=True)


def train_popeve(tup):  # Just passing in the tuple because too lazy to use starmap unordered
    training_data_file, protein_id, unique_id = tup
    if not silent:
        print(f"Starting {protein_id}")
    try:
        s3_cp_file(f"s3://markslab-private/popEVE/{training_data_file}", f"/tmp/data/{training_data_file}", silent=silent)
    except subprocess.CalledProcessError:
        print(f"Skipping {protein_id}")
        return
    
    # If we need to suppress outputs, we can use this
    # with suppress_stdout_stderr():
    #     pass
    
    losses_and_scales_path = losses_and_scales_directory + unique_id + '_loss_lengthscale.csv'
    scores_path = scores_directory + unique_id + '_scores.csv'
    
    training_data_df = pd.read_csv(f"/tmp/data/{training_data_file}")
    train(training_data_df=training_data_df, 
          protein_id=protein_id, 
          unique_id=unique_id, 
          losses_and_scales_path=losses_and_scales_path, 
          scores_path=scores_path,
          states_directory=model_states_directory)

    # Copy results to s3
    s3_cp_file(losses_and_scales_path, f"s3://markslab-private/popEVE/results/{run_name}/losses_and_lengthscales/{unique_id}_loss_lengthscale.csv", silent=silent)
    s3_cp_file(scores_path, f"s3://markslab-private/popEVE/results/{run_name}/scores/{unique_id}_scores.csv", silent=silent)
    
    # TODO could also just copy over the final checkpoint (unique_id + '_model_final.pth')
    checkpoints_to_copy = [f for f in os.listdir(model_states_directory) if unique_id in f and f.endswith(".pth")]
    for checkpoint_filename in checkpoints_to_copy:
        s3_cp_file(f"{model_states_directory}/{checkpoint_filename}", f"s3://markslab-private/popEVE/results/{run_name}/model_states/{checkpoint_filename}", silent=silent)
    
    # Clean up afterwards
    os.remove(f"/tmp/data/{training_data_file}")
    os.remove(losses_and_scales_path)
    os.remove(scores_path)
    for checkpoint_filename in checkpoints_to_copy:
        os.remove(f"{model_states_directory}/{checkpoint_filename}")
    
    print(unique_id)
    
    return unique_id


# Simple: One parallelisation run for a given setup (so only one results folder)
# Iterate over dataframe, get tuples, pass to function
num_cpus = len(os.sched_getaffinity(0))

print("Mapping file head:", mapping_df.head())

with Pool(num_cpus) as pool:
    results = tqdm(pool.imap_unordered(train_popeve, ((row.S3, row.protein_id, row.unique_id) for row in mapping_df.itertuples()), chunksize=1), total=len(mapping_df))

all_unique_ids_successful = [r for r in results if r]
print(f"{len(all_unique_ids_successful)} / {len(mapping_df)} successful")
# print(all_unique_ids_successful)  # Around 20k max
with open(f"{run_name}_successful.txt", "w") as f:
    f.write("\n".join(all_unique_ids_successful))
s3_cp_file(f"{run_name}_successful.txt", f"s3://markslab-private/popEVE/results/{run_name}/successful.txt", silent=True)
    

# for run_name in run_names:
#     dest_name = run_name + "_m0"
#     result = process_map(
#         partial(get_aln_stats, run_name, dest_name),
#         (row.protein_name for row in prio_df.itertuples()),
#         max_workers=len(os.sched_getaffinity(0)) // 8,
#         chunksize=10,
#         total=len(prio_df),
#     )
#     result_df = pd.DataFrame([r for r in result if r])
#     # result_df.to_csv(f"{run_name}_aln_stats.csv", index=False)
#     result_df.to_csv(f"{dest_name}_aln_stats.csv", index=False)
#     s3_cp_file(f"./{dest_name}_aln_stats.csv", "s3://markslab-private/eve/indels/data/mappings/")
