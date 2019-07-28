#!/bin/bash

# Job name
#SBATCH --job-name=two-dim-densities-oaccd-6-132-0.01
#
# Project
#SBATCH --account=nn2977k
#
# Wall clock limit
#SBATCH --time=02:00:00
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

## Run commands
cd $SCRATCH
export OMP_NUM_THREADS=4
python -u oaccd_n=6_l=132_omega=0.01.py | tee $SCRATCH/dat/oaccd_n=6_l=132_omega=0.01.log
