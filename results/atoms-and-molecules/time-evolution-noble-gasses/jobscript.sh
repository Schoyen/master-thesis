#!/bin/bash

# Job name
#SBATCH --job-name=ne-aug-ccpvtz
#
# Project
#SBATCH --account=nn2977k
#
# Wall clock limit
#SBATCH --time=168:00:00
#
# Max memory usage per core (MB)
#SBATCH --mem-per-cpu=15G
#
# Number of CPU's/processes
#SBATCH --cpus-per-task=4

## Set up job environment
source /cluster/bin/jobsetup
module purge
set -o errexit

## Copy files to work directory
cp $SUBMITDIR/*.py $SCRATCH

## Create directory for data
mkdir $SCRATCH/dat

## Mark outfiles for copying
cleanup "cp $SCRATCH/dat/* $SUBMITDIR/dat/"

## Run commands
cd $SCRATCH
export OMP_NUM_THREADS=4
python ne_aug-ccpvtz.py
