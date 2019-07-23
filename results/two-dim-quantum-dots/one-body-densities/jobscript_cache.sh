#!/bin/bash

# Job name
#SBATCH --job-name=two-dim-cache
#
# Project
#SBATCH --account=nn2977k
#
# Wall clock limit
#SBATCH --time=04:00:00
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

## Copy cached Coulomb elements to work directory
cp $SUBMITDIR/*.npy $SCRATCH

## Create directory for data
mkdir $SCRATCH/dat

## Mark outfiles for copying
cleanup "cp $SCRATCH/dat/* $SUBMITDIR/dat/"
cleanup "cp $SCRATCH/*.npy $SUBMITDIR/"

## Run commands
cd $SCRATCH
export OMP_NUM_THREADS=4
python -u setup_large_system.py | tee $SCRATCH/dat/setup_large_system.log
