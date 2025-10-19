#!/bin/bash

for i in {0..100}
do
    echo $i 
    sbatch patch_diagnosis.slurm $i
done
