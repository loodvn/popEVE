#!/bin/bash
# Setup the EC2 instance for popEVE (Ubuntu)

set -e
# Install tmux
sudo apt install tmux
# Install Miniforge
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh -b  # -b for batch (non-interactive) install

source ~/.bashrc  # To get Mamba in the namespace

# Install GPyTorch
mamba create -n popeve gpytorch pandas tqdm -c gpytorch  # Note: No version here 
mamba activate popeve

# AWS CLI
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# Test: aws s3 ls s3://markslab/

aws configure set default.s3.max_concurrent_requests 100