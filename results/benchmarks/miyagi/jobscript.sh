#!/bin/bash

# Job name
#SBATCH --job-name=benchmark-thesis
#
# Project
#SBATCH --account=nn2977k
#
# Wall clock limit
#SBATCH --time=02:00:00
#
# Max memory usage per core (MB)
#SBATCH --mem-per-cpu=1G
#
# Max memory usage
#SBATCH --cpus-per-task=8

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
make -j2
