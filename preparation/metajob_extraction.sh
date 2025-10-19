#!/bin/bash

for i in {0..499}
do
    echo $i 
    sbatch patch_extraction.slurm $i
done
