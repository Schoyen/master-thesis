#!/bin/bash

# Job name
#SBATCH --job-name=two-dim-densities-ccsd
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

## Run commands
cd $SCRATCH
export OMP_NUM_THREADS=4
#python -u ccsd_n=2_l=132_omega=0.1.py | tee $SCRATCH/dat/ccsd_n=2_l=132_omega=0.1.log
python -u ccsd_n=2_l=132_omega=0.01.py | tee $SCRATCH/dat/ccsd_n=2_l=132_omega=0.01.log
#python -u ccsd_n=2_l=132_omega=0.28.py | tee $SCRATCH/dat/ccsd_n=2_l=132_omega=0.28.log

#python -u ccsd_n=6_l=132_omega=0.01.py | tee $SCRATCH/dat/ccsd_n=6_l=132_omega=0.01.log
#python -u ccsd_n=6_l=132_omega=0.1.py | tee $SCRATCH/dat/ccsd_n=6_l=132_omega=0.1.log
#python -u ccsd_n=6_l=132_omega=0.28.py | tee $SCRATCH/dat/ccsd_n=6_l=132_omega=0.28.log
#
#python -u ccsd_n=12_l=132_omega=0.01.py | tee $SCRATCH/dat/ccsd_n=12_l=132_omega=0.01.log
#python -u ccsd_n=12_l=132_omega=0.1.py | tee $SCRATCH/dat/ccsd_n=12_l=132_omega=0.1.log
#python -u ccsd_n=12_l=132_omega=0.28.py | tee $SCRATCH/dat/ccsd_n=12_l=132_omega=0.28.log
#
#python -u ccsd_n=20_l=132_omega=0.01.py | tee $SCRATCH/dat/ccsd_n=20_l=132_omega=0.01.log
#python -u ccsd_n=20_l=132_omega=0.1.py | tee $SCRATCH/dat/ccsd_n=20_l=132_omega=0.1.log
#python -u ccsd_n=20_l=132_omega=0.28.py | tee $SCRATCH/dat/ccsd_n=20_l=132_omega=0.28.log
