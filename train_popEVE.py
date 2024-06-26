# TODO Have to stop as Luis is awake. 
# What is currently here is the code from the tutorial
# You need to now merge this with other cells from the tutorial to do with handling the data, cuda etc.
# You need to also include the bits from the old script that are not present in the turorial
# You need to account for changes of file names etc

import argparse
import torch
import pandas as pd
import gpytorch
from tqdm import trange
import time

from popEVE.popEVE import PGLikelihood, GPModel
from utils.helpers import get_training_and_holdout_data_from_processed_file, get_scores

if torch.cuda.is_available():
    print("GPU is available!")
else:
    print("GPU is not available. CPU will be used.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

def parse_args():
    parser = argparse.ArgumentParser(description='VAE')
    parser.add_argument('--mapping_file', type=str, help='List of genes and corresponding training data file path')
    parser.add_argument('--gene_index', type=int, help='Row index of gene in gene_list')
    parser.add_argument('--losses_dir', type=str, help='File path for saving losses and lengthscales')
    parser.add_argument('--scores_dir', type=str, help='File path for saving scores')
    parser.add_argument('--model_states_dir', type=str, help='File path for model states')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')  
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--skip_slow', type=int, default=0, help='Skip any processes that would take more than X hours to run.')
    # TODO add nogpu or something to ensure we're being explicit about GPU
    args = parser.parse_args()
    return args

def main(mapping_file, gene_index, losses_dir, scores_dir, model_states_dir, seed=42, file_path_column_name="file_path", debug=False, skip_slow=0):
    df_mapping = pd.read_csv(mapping_file)
    protein_id = df_mapping['protein_id'][gene_index]
    unique_id = df_mapping['unique_id'][gene_index]
    training_data_df = pd.read_csv(df_mapping[file_path_column_name][gene_index])
    
    losses_and_scales_directory = losses_dir
    scores_directory = scores_dir
    states_directory = model_states_dir
    
    losses_and_scales_path = losses_and_scales_directory + unique_id + '_loss_lengthscale.csv'
    scores_path = scores_directory + unique_id + '_scores.csv'
    
    success, reason = train(training_data_df, protein_id, unique_id, losses_and_scales_path, scores_path, states_directory, seed, debug, skip_slow)
    if not success:
        if debug:
            print(reason)
        # Write out a .failed file (basically 'touch')
        with open(f"{scores_directory}/{unique_id}.failed", "a") as f:
            f.write(reason) # Useful to record so that we can go back and check specifically for this unique ID
    print("Done")
    
    
def train(training_data_df, protein_id, unique_id, losses_and_scales_path, scores_path, states_directory, seed=42, debug=False, skip_slow=0, device=device):
    train_x, train_y, train_variants, heldout_x, heldout_y, heldout_variants, X_min, X_max = get_training_and_holdout_data_from_processed_file(training_data_df, device = device)

    unique_train_output = train_y.unique(return_counts = True)
    unique_test_output = heldout_y.unique(return_counts = True)
    print(f"Protein ID = {protein_id}")
    print(f"Train: y unique = {unique_train_output[0]}, y counts = {unique_train_output[1]}")
    print(f"Holdout: y unique = {unique_test_output[0]}, y counts = {unique_test_output[1]}")

    # Initialize the model with M = 20 inducing points
    M = 20
    inducing_points = torch.linspace(0, 1, M, dtype=train_x.dtype, device=train_x.device).unsqueeze(-1)
    model = GPModel(inducing_points=inducing_points)

    # Initialize the lengthscale of the base kernel in the covariance module
    model.covar_module.base_kernel.initialize(lengthscale=0.2)

    # Initialize the likelihood
    likelihood = PGLikelihood()

    # Move the model and likelihood to the specified device (e.g., GPU)
    model = model.to(device)
    likelihood = likelihood.to(device)

    # Set the learning rates for the NGD optimizer and the Adam optimizer
    lr1 = 0.1
    lr2 = 0.05

    # Initialize the NGD optimizer for variational parameters
    variational_ngd_optimizer = gpytorch.optim.NGD(model.variational_parameters(), num_data=train_y.size(0), lr=lr1)

    # Initialize the Adam optimizer for hyperparameters
    hyperparameter_optimizer = torch.optim.Adam([
        {'params': model.hyperparameters()},
        {'params': likelihood.parameters()},
    ], lr=lr2)

    # Set the model and likelihood to training mode
    model.train()
    likelihood.train()

    # Initialize the variational ELBO (Evidence Lower Bound) object
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y.size(0))

    # Set the number of training epochs
    num_epochs = 6000

    # Use trange for a progress bar over the epochs
    epochs_iter = trange(num_epochs)

    # Lists to store losses and lengthscales for analysis
    losses = []
    lengthscales = []
    
    skip_duration = skip_slow * 60*60 / num_epochs   # e.g. if 1 hour is the minimum, then skip all processes that have a speed of > 3600/6000 = 0.6 seconds per epoch

    # Training loop
    for i in epochs_iter:
        if skip_slow > 0 and i == 1:
            start_time = time.perf_counter()
        # Perform NGD step to optimize variational parameters and hyperparameters
        variational_ngd_optimizer.zero_grad()
        hyperparameter_optimizer.zero_grad()

        # Forward pass to compute the output and calculate the loss
        output = model(train_x)
        loss = -mll(output, train_y)

        # Backward pass to compute gradients and perform optimization steps
        loss.backward()
        variational_ngd_optimizer.step()
        hyperparameter_optimizer.step()

        # Store the loss and lengthscales for later analysis
        losses.append(loss.item())
        lengthscale = model.covar_module.base_kernel.lengthscale.item()
        lengthscales.append(lengthscale)

        if skip_duration > 0 and i == 1:
            elapsed = time.perf_counter() - start_time
            if elapsed > skip_duration:
                print(f"{protein_id} too slow. Time elapsed in second loop = {elapsed:.2f}s > {skip_duration:.2f}s (i.e. would take over {skip_slow} hours). Quitting.")
                return False, f"Slow.{elapsed:.2f}"
        # Save model every 1000 epochs
        if i % 1000 == 0 and i != 0:
            model_state_path = f"{states_directory}/{unique_id}_model_{str(i)}.pth"
            torch.save(model.state_dict(), model_state_path)

    # Save final model
    model_final_checkpoint = f"{states_directory}/{unique_id}_model_final.pth"
    torch.save(model.state_dict(), model_final_checkpoint)

    # Save loss and correlation length info
    pd.DataFrame({'loss': losses, 'lengthscale': lengthscales}).to_csv(losses_and_scales_path, index = False)

    # Compute scores for every possible single amino acid substitution
    scores_df = get_scores(model, train_x, train_variants, sample_size = 10**3)
    scores_df.to_csv(scores_path, index = False)
    
    return True, ""

if __name__=='__main__':
    args = parse_args()
    main(**args.__dict__)
