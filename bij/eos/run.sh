#!/bin/bash

#SBATCH --job-nam=f6
#SBATCH --account=etna
#SBATCH --ntasks=192
#SBATCH --partition=etna
#SBATCH --time=48:0:0
#SBATCH --signal=B:10@300

mpirun -n 192 npt fluid*.conf -s 1000000 -o 1 -v 0  --snapshot 0 
