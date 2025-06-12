__title__ = "converts a netcdf with seperate variables to a UM ancil using a template"
__version__ = "2023-03-29"
__author__ = "Mathew Lipson"
__email__ = "mathew.lipson@bom.gov.au"
__institution__ = "Bureau of Meteorology"

print('''
This script takes the compressed xarray netcdf and converts it to a UM ancil using MULE 
(as NCI ANTS can not save um files for some reason).

on gadi use:
    module use /g/data/hh5/public/modules
    module load conda/analysis3-22.10
''')

import os
import iris
import ants
import numpy as np
import xarray as xr
import mule
import argparse
# silence iris/ants warnings
import warnings
warnings.filterwarnings("ignore")

import common_functions as cf
import importlib
importlib.reload(cf)

home = os.getenv("HOME")
parser = argparse.ArgumentParser(description='converts a netcdf with seperate variables to a UM ancil using a template')
parser.add_argument('--template', dest='template_fpath', help='template: path to qrparm.veg.frac* template', default=f'{home}/cylc-run/au-L/share/data/ancils/au-L/auL_0p04/qrparm.veg.frac_cci_pre_c4')
parser.add_argument('--source', dest='source_fpath', help='source: path to ESA CCIv2_WC regridded file', default=f'{home}/cylc-run/au-L/share/data/ancils/au-L/auL_0p04/working/cciv2_wc_pre_c4_regridded.nc')
parser.add_argument('--output', dest='output_dir', help='output directory', default=f'{home}/cylc-run/au-L/share/data/ancils/au-L/auL_0p04/working')
args = parser.parse_args()

template_fpath = args.template_fpath
source_fpath = args.source_fpath
output_dir = args.output_dir

def main():

    print('loading ancil template')
    template = get_template(template_fpath)

    print(f'opening source data from {source_fpath}')
    orig_ds = xr.open_dataset(source_fpath)
    orig_ds = orig_ds.where(orig_ds<=1)
    updated = orig_ds[['broad_leaf', 'needle_leaf', 'c3_grass', 'c4_grass', 'shrub', 'urban', 'lake', 'soil', 'ice']]
    updated = updated.interp_like(template,method='nearest').combine_first(template)

    print('ensuring updated matches template')
    total_template,total_updated = get_checksum(template), get_checksum(updated)
    xr.testing.assert_equal(updated[['latitude','longitude']], template[['latitude','longitude']])
    xr.testing.assert_allclose(total_updated, total_template, 1E-6)

    #### WRITE ####
    print('saving to UM ancil (with MULE) and netCDF (with ANTS)')
    lct_file = os.path.basename(template_fpath)  
    updated_fpath = f'{output_dir}/{lct_file}_cci_wc'
    update_ancil_with_mule(updated,template_fpath,updated_fpath)
    source_cubes = ants.load_cube(updated_fpath,constraint='m01s00i216')
    ants.save(source_cubes, updated_fpath+'.nc',saver='nc')

    return

def get_checksum(da,rtol=1E-6):
    '''check variables sum to 1'''

    # test geoscape classess sum to 1
    if isinstance(da,xr.core.dataarray.DataArray):
        total = da.sum(axis=0,skipna=False)
    elif isinstance(da,xr.core.dataset.Dataset):
        total = da.to_array().sum(axis=0,skipna=False)
    else:
        print('type not known')

    # assert all nonull values are close to 1
    np.testing.assert_allclose(total.fillna(1), 1, rtol)
    print('sums to 1')

    return total

def update_ancil_with_mule(ds,template_fpath,updated_fpath):
    '''
    Description:
        Updates an ancil file (e.g. land.frac) with new values from the provided numpy array
        ds must have the same shape and nan must be filled with the fill_val from the template, can be: 
            - xarray Dataset
            - xarray DataArray
            - iris Cube
   Arguments:
        arr: numpy.array object with the same shape as the template
        template_fpath, updated_fpath: paths to the template and the updated file
    '''

    cb = iris.load_cube(template_fpath)

    if np.ma.is_masked(cb.data):
        if isinstance(ds,xr.core.dataarray.DataArray):
            arr = ds.fillna(value=cb.data.fill_value).values
        elif isinstance(ds,xr.core.dataset.Dataset):
            arr = ds.to_array().fillna(value=cb.data.fill_value).values
        elif isinstance(ds,iris.cube.Cube):
            arr = ds.data.data
    else:
        arr = ds.to_array().values

    print(f'loading {template_fpath} as template')
    ancil = mule.AncilFile.from_file(template_fpath)
    # ancil = mule.UMFile.from_file(template_fpath)  # no validation

    for i,field in enumerate(ancil.fields):
        if field.lbuser4 == 216:
            print(f'updating field {i}')
            array_provider = mule.ArrayDataProvider(arr[i,:,:])
            ancil.fields[i].set_data_provider(array_provider)

    print(f'saving updated ancil to {updated_fpath}')
    try:
        ancil.to_file(updated_fpath)
    except Exception as e:
        print(e)
        print('WARNING: MULE validation being disabled')
        # Scott Wales suggestion to turn of validation for writing Ancils
        ancil.validate = lambda *args, **kwargs: True
        ancil.to_file(updated_fpath)

    print('ancil updated and saved')

    return

def get_template(template_fpath):

    if os.path.exists(template_fpath):
        template = cf.convert_ancil_to_ds(template_fpath,title='template for fractional land cover')

        if 'grid_latitude' in template.coords.keys():
            template = template.rename(grid_latitude='latitude',grid_longitude='longitude')
    else:
        raise Exception(f'{os.path.basename(template_fpath)} not found. Has suite been run up until ancil_lct_postproc_c4?')
    
    return template

if __name__ == '__main__':
    main()
