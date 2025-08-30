#!/bin/bash
#PBS -q normal
#PBS -l ncpus=24
#PBS -l mem=96GB
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/ce10+gdata/ce10+scratch/public+scratch/pu02+gdata/gb02+gdata/ob53+gdata/ra22+gdata/fy29+gdata/xp65
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -P fy29

module purge
module use /g/data/xp65/public/modules
module load conda/analysis3

python /home/561/mjl561/git/RNS_NA_trials/plotting/plot_rain_accum_single.py