#!/bin/bash

### Full set
sbatch xgb_model_training.slurm "env_rs" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/global/full/xgb/"

### Full set - Abio
sbatch xgb_model_training.slurm "rs" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/global/ablation_env/xgb/"

### Full set - RSEBV
sbatch xgb_model_training.slurm "env" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/global/ablation_rs/xgb/"

### Full set - MSI
sbatch xgb_model_training.slurm "env_rs" "none" "dofa_s1" "habitat_models/global/ablation_msi/xgb/"

### Full set - SAR
sbatch xgb_model_training.slurm "env_rs" "ssl4eo_vit_moco" "none" "habitat_models/global/ablation_sar/xgb/"