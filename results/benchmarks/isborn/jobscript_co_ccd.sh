#!/bin/bash

# Job name
#SBATCH --job-name=co-isborn-ccd
#
# Project
#SBATCH --account=nn2977k
#
# Wall clock limit
#SBATCH --time=168:00:00
#
# Max memory usage per core (MB)
#SBATCH --mem-per-cpu=4G
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
#python run_ccd.py co sto-3g both
#python run_ccd.py co sto-3g up

#python run_ccd.py co 6-31gss both
python run_ccd.py co 6-31gss up
