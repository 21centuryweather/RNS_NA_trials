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
import xarray as xr
import matplotlib.pyplot as plt
import argparse

import common_functions as cf
import importlib
importlib.reload(cf)

home = os.getenv('HOME')
parser = argparse.ArgumentParser(description='plot land cover ancillary')
parser.add_argument('--cciv2_fpath', dest='cciv2_fpath', help='path to CCIv2', default=f'{home}/cylc-run/SY_ancil/share/data/ancils/SY/SY_333/qrparm.veg.frac_cci_pre_c4')
parser.add_argument('--updated_fpath', dest='updated_fpath', help='path to updated CCIv2+WC', default='')
parser.add_argument('--orig_fpath', dest='orig_fpath', help='path to original ancil CCI', default='')
parser.add_argument('--domain_name', dest='domain_name', help='domain name', default=f'Oceania')
parser.add_argument('--plotpath', dest='plotpath', help='plot path', default='./')
args = parser.parse_args()

def plot_comparison(subdomain, bounds):

    # ensure output directory exists
    if not os.path.exists(args.plotpath):
        print(f'making directory {args.plotpath}')
        os.makedirs(args.plotpath)
    
    print('opening and subsetting data')
    cciv2 = xr.open_dataset(args.cciv2_fpath)
    cciv2.attrs['title'] = f'{args.domain_name} land cover ancillary (CCIv2)'
    cciv2_wc = cf.convert_ancil_to_ds(args.updated_fpath, title=f'{args.domain_name} land cover ancillary (CCIv2+WorldCover)')
    orig = cf.convert_ancil_to_ds(args.orig_fpath,title=f'{args.domain_name} land cover ancillary (CCI)')

    cciv2 = get_subset(cciv2,bounds)
    cciv2_wc = get_subset(cciv2_wc,bounds)
    orig = get_subset(orig,bounds)
    
    # remove ANTS fill if necessary
    cciv2 = cciv2.where(cciv2<1E3)
    
    # combine tiles
    cciv2_wc = cf.combine_tiles(cciv2_wc)
    orig = cf.combine_tiles(orig)
    
    #### PLOT ####
    def _plots(ds,name,title,source):
        print(f'plotting {name}')
        cmap = 'viridis'
        dsp = cf.combine_tiles(ds.copy())
        plot_order, ncols=['broad_leaf', 'needle_leaf', 'grass', 'shrub', 'urban', 'lake', 'soil', 'ice'], 4
        fig = cf.plot_landcover_fractions(dsp,title,source,plot_order,ncols=ncols,cmap=cmap)
        fig.savefig(f'{args.plotpath}/{args.domain_name}_julesPFTs_{subdomain}_{name}.png', dpi=300, bbox_inches='tight')
        fig = cf.plot_tile(dsp,title,source,'urban',bounds,cmap)
        fig.savefig(f'{args.plotpath}/{args.domain_name}_jules_urban_{subdomain}_{name}.png', dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    # _plots(cciv2,name='cciv2',title=cciv2.title,source=args.cciv2_fpath)
    _plots(cciv2_wc,name='cciv2_wc',title=cciv2_wc.title,source=args.updated_fpath)
    _plots(orig,name='orig',title=orig.title,source=args.orig_fpath)
    print(f'plots completed, see {args.plotpath}')

    return

def get_subset(ds,bounds):

    xmin, ymin, xmax, ymax = bounds

    ds = ds.sel(latitude=slice(ymax,ymin), longitude=slice(xmin,xmax))
    # check if lats need to be reversed
    if len(ds.latitude)==0:
        ds = ds.sel(latitude=slice(ymin,ymax), longitude=slice(xmin,xmax))

    return ds

if __name__ == '__main__':

    subdomain, bounds = 'full', (None,None,None,None)
    plot_comparison(subdomain, bounds)

    # domain, bounds = 'sydney', (148, -35, 152, -31)
    # plot_comparison(subdomain, bounds)



