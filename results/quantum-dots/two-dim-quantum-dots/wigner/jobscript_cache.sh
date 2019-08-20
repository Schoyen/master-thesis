#!/bin/bash

# Job name
#SBATCH --job-name=two-dim-cache-210
#
# Project
#SBATCH --account=nn2977k
#
# Wall clock limit
#SBATCH --time=40:00:00
#
# Max memory usage per core (MB)
#SBATCH --mem-per-cpu=15300
#
# Number of CPU's/processes
#SBATCH --cpus-per-task=4

## Set up job environment
source /cluster/bin/jobsetup
module purge
set -o errexit

## Copy files to work directory
cp $SUBMITDIR/*.py $SCRATCH

## Mark outfiles for copying
cleanup "cp $SCRATCH/*.npy $SUBMITDIR/"

## Run commands
cd $SCRATCH
export OMP_NUM_THREADS=4
python setup_large_system.py 210
