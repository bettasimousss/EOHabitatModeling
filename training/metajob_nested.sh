#!/bin/bash

### Abio
sbatch xgb_model_training.slurm "env" "MA2" "none" "none" "habitat_models/nested/MA2/abio/xgb/"
sbatch xgb_model_training.slurm "env" "N" "none" "none" "habitat_models/nested/N/abio/xgb/"
sbatch xgb_model_training.slurm "env" "P" "none" "none" "habitat_models/nested/P/abio/xgb/"
sbatch xgb_model_training.slurm "env" "Q" "none" "none" "habitat_models/nested/Q/abio/xgb/"
sbatch xgb_model_training.slurm "env" "R" "none" "none" "habitat_models/nested/R/abio/xgb/"
sbatch xgb_model_training.slurm "env" "S" "none" "none" "habitat_models/nested/S/abio/xgb/"
sbatch xgb_model_training.slurm "env" "T" "none" "none" "habitat_models/nested/T/abio/xgb/"
sbatch xgb_model_training.slurm "env" "U" "none" "none" "habitat_models/nested/U/abio/xgb/"
sbatch xgb_model_training.slurm "env" "V" "none" "none" "habitat_models/nested/V/abio/xgb/"

### Abio+RS
sbatch xgb_model_training.slurm "env_rs" "MA2" "none" "none" "habitat_models/nested/MA2/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "N" "none" "none" "habitat_models/nested/N/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "P" "none" "none" "habitat_models/nested/P/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "Q" "none" "none" "habitat_models/nested/Q/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "R" "none" "none" "habitat_models/nested/R/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "S" "none" "none" "habitat_models/nested/S/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "T" "none" "none" "habitat_models/nested/T/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "U" "none" "none" "habitat_models/nested/U/abio_rs/xgb/"
sbatch xgb_model_training.slurm "env_rs" "V" "none" "none" "habitat_models/nested/V/abio_rs/xgb/"


### Abio+RS+MSI
sbatch xgb_model_training.slurm "env_rs" "MA2" "ssl4eo_vit_moco" "none" "habitat_models/nested/MA2/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "N" "ssl4eo_vit_moco" "none" "habitat_models/nested/N/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "P" "ssl4eo_vit_moco" "none" "habitat_models/nested/P/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "Q" "ssl4eo_vit_moco" "none" "habitat_models/nested/Q/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "R" "ssl4eo_vit_moco" "none" "habitat_models/nested/R/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "S" "ssl4eo_vit_moco" "none" "habitat_models/nested/S/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "T" "ssl4eo_vit_moco" "none" "habitat_models/nested/T/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "U" "ssl4eo_vit_moco" "none" "habitat_models/nested/U/abio_rs_msi/xgb/"
sbatch xgb_model_training.slurm "env_rs" "V" "ssl4eo_vit_moco" "none" "habitat_models/nested/V/abio_rs_msi/xgb/"

### Abio+RS+MSI+SAR
sbatch xgb_model_training.slurm "env_rs" "MA2" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/MA2/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "N" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/N/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "P" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/P/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "Q" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/Q/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "R" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/R/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "S" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/S/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "T" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/T/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "U" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/U/abio_rs_msi_sar/xgb/"
sbatch xgb_model_training.slurm "env_rs" "V" "ssl4eo_vit_moco" "dofa_s1" "habitat_models/nested/V/abio_rs_msi_sar/xgb/"
