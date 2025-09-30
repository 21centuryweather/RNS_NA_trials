#!/bin/bash
#PBS -q hugemem
#PBS -l ncpus=12
#PBS -l mem=384GB
#PBS -l walltime=04:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/ce10+gdata/ce10+scratch/public+scratch/pu02+gdata/gb02+gdata/ob53+gdata/ra22+gdata/fy29+gdata/xp65
#PBS -l wd
#PBS -l jobfs=4GB
#PBS -P fy29

module purge
module use /g/data/hh5/public/modules
module load conda/analysis3

python /home/561/mjl561/git/RNS_NA_trials/plotting/plot_moist_static_energy.py