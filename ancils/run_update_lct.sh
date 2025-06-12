#!/bin/bash
#PBS -q normal
#PBS -l ncpus=8
#PBS -l mem=32GB
#PBS -l walltime=01:00:00
#PBS -l storage=gdata/hh5+gdata/access+gdata/dp9+scratch/dp9+scratch/fy29
#PBS -l wd
#PBS -l jobfs=10GB
#PBS -P fy29

set -e

# suite information
PROJ=fy29                    # your project code
SUITE_ID='ancils_NA_trials'            # your suite ID (e.g. 'u-bu503')
REGION='NA'             # your region (e.g. 'ACCESS-A')

REGPATH=/scratch/${PROJ}/${USER}/cylc-run/${SUITE_ID}/share/data/ancils/${REGION}
SRC_DIR=/home/561/mjl561/git/au_ancils/update_lct
SRC_DIR=/home/561/mjl561/git/RNS_NA_trials/preproc_lct/update_lct


# loop through resolution folders
for RES in `find $REGPATH -maxdepth 1 -mindepth 1 -type d -print0 | xargs -0 basename -a`
do
    echo "Processing $RES"

    # if RES contains 'era5', then skip
    if [[ $RES == *"era5"* ]]; then
        echo "Skipping $RES"
        continue
    fi

    # paths
    ANCIL_DIR=$REGPATH/$RES
    WORKING_DIR=$ANCIL_DIR/working
    ANTS_SRC=$HOME/cylc-run/$SUITE_ID/src/ants/bin
    CCIV2_PATH='/g/data/dp9/mjl561/au_ancils_source/Oceania_CCIv2_with_WorldCover_urban_300m.nc'

    # inputs etc
    MASK_FILE=qrparm.mask_cci
    LCT_FILE=qrparm.veg.frac_cci_pre_c4
    ANCIL_CONFIG=$SRC_DIR/ants_config.conf
    CCI_OUTPUT=cciv2_wc_pre_c4_regridded.nc

    module purge
    module use /g/data/access/ngm/modules
    module load ants/0.18

    # regrid CCIv2+WC (300m) to template resolution using ANTS with AreaWeighted method (slower but more accurate)
    python $ANTS_SRC/ancil_general_regrid.py $CCIV2_PATH \
        --ants-config $ANCIL_CONFIG \
        --target-lsm $ANCIL_DIR/$MASK_FILE \
        --output $WORKING_DIR/$CCI_OUTPUT

    module purge
    module use /g/data/hh5/public/modules
    module load conda/analysis3-22.10

    # convert netcdf to um format based on template
    python $SRC_DIR/netcdf_2ancil.py \
        --template $ANCIL_DIR/$LCT_FILE \
        --source $WORKING_DIR/$CCI_OUTPUT \
        --output $WORKING_DIR

    # [if it doesn't exist] move original LCT_FILE and symlink new (only .nc used by the suite)
    if [[ ! -f ${WORKING_DIR}/${LCT_FILE}.original.nc ]]; then
        echo moving and symlinking $LCT_FILE
        mv ${ANCIL_DIR}/${LCT_FILE}.nc ${WORKING_DIR}/${LCT_FILE}.original.nc
        cd $ANCIL_DIR
        ln -sf ./working/${LCT_FILE}_cci_wc.nc $LCT_FILE.nc
    else
        echo ${LCT_FILE}.original.nc already exits, no symlink created
    fi

    # plot comparison
    python $SRC_DIR/plot_comparison.py \
        --cciv2_fpath $WORKING_DIR/$CCI_OUTPUT \
        --updated_fpath $WORKING_DIR/${LCT_FILE}_cci_wc.nc \
        --orig_fpath $WORKING_DIR/$LCT_FILE.original.nc \
        --domain_name $RES \
        --plotpath $WORKING_DIR

done

echo "done!"

