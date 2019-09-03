#!/bin/bash

# Job name
#SBATCH --job-name=ion-20-hf
#
# Project
#SBATCH --account=nn2977k
#
# Wall clock limit
#SBATCH --time=16:00:00
#
# Max memory usage per core (MB)
#SBATCH --mem-per-cpu=4G
#
# Number of CPU's/processes
#SBATCH --cpus-per-task=8

## Set up job environment
source /cluster/bin/jobsetup
module purge
set -o errexit

## Copy files to work directory
cp $SUBMITDIR/*.py $SCRATCH
cp $SUBMITDIR/Makefile $SCRATCH

## Create directory for data
mkdir $SCRATCH/dat

## Mark outfiles for copying
cleanup "cp $SCRATCH/dat/* $SUBMITDIR/dat/"

## Run commands
cd $SCRATCH
export OMP_NUM_THREADS=8
#python ccd_miyagi.py 20
#python ccd_miyagi.py 30
#python ccd_miyagi.py 36
#python ccd_miyagi.py 40

python hf_miyagi.py 20
#python hf_miyagi.py 30
#python hf_miyagi.py 36
#python hf_miyagi.py 40
