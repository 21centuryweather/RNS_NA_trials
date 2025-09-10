#!/bin/bash
#PBS -q normal
#PBS -l ncpus=12
#PBS -l mem=48GB
#PBS -l walltime=06:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/ce10+gdata/ce10+scratch/public+scratch/pu02+gdata/gb02+gdata/ob53+gdata/ra22+gdata/fy29+gdata/xp65+gdata/zv2
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -P fy29
#PBS -M m.lipson@unsw.edu.au
#PBS -N run_agcd_all


#########################################################
# Run AGCD comparison plots for multiple years
# 
# This script loops through years 2014-2020, running
# plot_agcd.py for each year with date ranges from
# December 1st to February 28th of the following year.
#
# Author: Mathew Lipson <m.lipson@unsw.edu.au>
# Date: 2025-09-10
#########################################################

# Set up environment
module use /g/data/hh5/public/modules
module load conda/analysis3

# Script directory
PYTHON_SCRIPT="/home/561/mjl561/git/RNS_NA_trials/plotting/plot_agcd.py"

echo "Starting AGCD comparison plots for years 2014-2020"
echo "Date range: Dec 1 - Feb 28 of following year"
echo "Script location: $PYTHON_SCRIPT"
echo ""

# Loop through years 2014-2020
for year in {2014..2020}; do
    
    # Calculate next year for February end date
    next_year=$((year + 1))
    
    # Set date range: Dec 1 of current year to Feb 28 of next year
    start_date="${year}-12-01"
    end_date="${next_year}-02-28"
    
    echo "=================================================="
    echo "Processing Year: $year"
    echo "Date range: $start_date to $end_date"
    echo "=================================================="
    
    # Run plot_agcd.py for both domains
    echo "Running AGCD comparison for year $year..."
    python "$PYTHON_SCRIPT" --year "$year" --sdate "$start_date" --edate "$end_date" --domain both
    
    # Check if the command succeeded
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed plots for year $year"
    else
        echo "✗ Error processing year $year"
    fi
    
    echo ""
    
done

echo "=================================================="
echo "AGCD comparison plotting completed for all years!"
echo "=================================================="
