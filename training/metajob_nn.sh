#!/bin/bash

sbatch mlp_model_training.slurm 0
sbatch mlp_model_training.slurm 1
sbatch mlp_model_training.slurm 2
sbatch mlp_model_training.slurm 3
sbatch mlp_model_training.slurm 4

sbatch tabnet_model_training.slurm 0
sbatch tabnet_model_training.slurm 1
sbatch tabnet_model_training.slurm 2
sbatch tabnet_model_training.slurm 3
sbatch tabnet_model_training.slurm 4