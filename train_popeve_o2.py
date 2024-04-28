#!/usr/bin/env python
# Copied from Aaron's get_alignment_stats.py in his EVE repo
import os
import sys
import argparse

import pandas as pd

from tqdm.auto import tqdm
from multiprocessing import Pool
import torch

sys.path.append("/home/ubuntu/popEVE/")
from train_popEVE import train

# Basically a parallel version of train_popEVE
def parse_args():
    parser = argparse.ArgumentParser(description='popEVE on O2')
    parser.add_argument('--mapping_file', type=str, help='List of genes and corresponding training data file path')
    parser.add_argument('--chunk_index', type=int, help='The starting index * script_chunk_size to process')
    parser.add_argument("--script_chunk_size", type=int, default=1, help="How many genes this script should process")
    parser.add_argument("--num_cpus", type=int, default=1, help="Number of CPUs to use")
    
    parser.add_argument('--losses_dir', type=str, help='File path for saving losses and lengthscales')
    parser.add_argument('--scores_dir', type=str, help='File path for saving scores')
    parser.add_argument('--model_states_dir', type=str, help='File path for model states')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--skip_slow', type=int, default=0, help='Skip any processes that would take more than X hours to run.')
    
    # TODO add nogpu or something to ensure we're being explicit about GPU
    args = parser.parse_args()
    return args
    
args = parse_args()
silent = not args.debug if args.debug else True  # This is the other way round basically
losses_and_scales_directory = args.losses_dir
scores_directory = args.scores_dir
model_states_directory = args.model_states_dir
overwrite = False

def train_popeve(tup):  # Just passing in the tuple because too lazy to use starmap unordered
    training_data_file, protein_id, unique_id = tup
    training_data_file_local_path = training_data_file
    
    # If we need to suppress outputs, we can use this
    # with suppress_stdout_stderr():
    #     pass
    
    losses_and_scales_path = losses_and_scales_directory + unique_id + '_loss_lengthscale.csv'
    scores_path = scores_directory + unique_id + '_scores.csv'
    
    if os.path.isfile(scores_path) and not overwrite:  # Could also check if the _final checkpoint exists
        if not silent:
            print(f"Skipping {unique_id}")
        result_dict = {"unique_id": unique_id, "result": True, "reason": "Skipped"}
        return result_dict
    
    
    if not silent:
        print("Copied file, reading in")
    
    training_data_df = pd.read_csv(training_data_file_local_path)
    if not silent:
        print("Read in file, training")
    
    torch.set_num_threads(1)  # Avoid CPU oversubscription
    
    result, reason = train(
        training_data_df=training_data_df, 
        protein_id=protein_id, 
        unique_id=unique_id, 
        losses_and_scales_path=losses_and_scales_path, 
        scores_path=scores_path,
        states_directory=model_states_directory,
        debug=not silent,
        skip_slow=args.skip_slow,  # Skip any runs that would take > 4 hours
        # TODO might have to set CPU device manually if we still have torch CPU clashes
    )
    
    if not result:
        if not silent:
            print(f"Failed {unique_id}: {reason}")
        # Just write out a .failed file recording the reason
        with open(f"{scores_directory}/{unique_id}.failed", "w") as f:
            f.write(reason)
    
    result_dict = {"unique_id": unique_id, "result": result, "reason": reason}
    
    return result_dict

# Simple: One parallelisation run for a given setup (so only one results folder)
# Iterate over dataframe, get tuples, pass to function
if __name__ == "__main__":
    # num_cpus = len(os.sched_getaffinity(0))
    num_cpus = args.num_cpus
    print(f"Using {num_cpus} CPUs.")

    mapping_df = pd.read_csv(args.mapping_file)
    
    # Get indices we need to operate over
    chunk_index = args.chunk_index
    print(f"Processing chunk {chunk_index}.")
    chunk_size = args.script_chunk_size
    subset_df = mapping_df.iloc[chunk_index*chunk_size:(chunk_index+1)*chunk_size]
    print(f"Processing {len(subset_df)} genes. Start of file:")
    print(subset_df.head())

    # Note: This might exist for overwrite-friendly runs. In that case we should append
    results_path = f"{scores_directory}/results_chunk{chunk_index}.csv"
    
    with Pool(num_cpus) as pool:
        results = tqdm(pool.imap_unordered(train_popeve, ((row.file_path, row.protein_id, row.unique_id) for row in subset_df.itertuples()), chunksize=1), total=len(mapping_df))
        # Write out as the results come in (has to stay within the context manager)
        for r in results:
            if not os.path.exists(results_path):
                print("Results path doesn't exist, creating and then appending")
                # Write header
                df_results = pd.DataFrame([r])
                df_results.to_csv(results_path, index=False, mode="a")
            else:
                df_results = pd.DataFrame([r])
                df_results.to_csv(results_path, index=False, mode="a", header=False)
                # Manually write out
                # with open(results_path, mode="a") as f:
                #     f.write(f'{",".join([r["unique_id"], r["result"], r["reason"]])}\n')
    

    print("Done")