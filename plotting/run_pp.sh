#!/bin/bash
#PBS -q normal
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/ce10+gdata/ce10+scratch/public+scratch/pu02+gdata/gb02+gdata/ob53+gdata/ra22+gdata/fy29+gdata/xp65
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -P fy29
#PBS -M m.lipson@unsw.edu.au
#PBS -N run_pp_2019

module purge
module use /g/data/hh5/public/modules
module load conda/analysis3-24.01
module load dask-optimiser

# Define years to process (modify this list as needed)

YEARS="2017"
YEARS="2016"
YEARS="2020"
YEARS="2015"
YEARS="2017"
YEARS="2018"
YEARS="2019"
YEARS="2014 2015 2016 2017 2018 2019 2020"
YEARS="2019"

# Automatically construct cylc_id arguments from years
CYLC_IDS=""
for year in $YEARS; do
    CYLC_IDS="$CYLC_IDS rns_ostia_NA_$year"
done

# Run the Python script with the constructed arguments
python /home/561/mjl561/git/RNS_NA_trials/plotting/plot_pp.py --cylc-id $CYLC_IDS
