__title__ = "contains functions common across files"
__version__ = "2023-06-11"
__author__ = "Mathew Lipson"
__email__ = "mathew.lipson@bom.gov.au"
__institution__ = "Bureau of Meteorology"

import os
import iris
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

# silence iris/ants warnings
import warnings
warnings.filterwarnings("ignore")


def playground():

    fname = '/home/561/mjl561/cylc-run/SY_ancil/share/data/ancils/SY/SY_333/working/qrparm.veg.frac_cci_pre_c4.original.nc'
    ds = convert_ancil_to_ds(fname,title='')

    return

################################
###### plotting functions ######

def convert_ancil_to_ds(fname,title=''):
    '''converts a UM file to an xarray dataset
    accounts for issues when ants is loaded which changes coordinate names'''

    cb = iris.load_cube(fname,constraint='m01s00i216')

    # check if ants has changed name of pseudo_level, if so revert to iris psuedo_level values
    if '_pseudo_level_order' in [coord.name() for coord in cb.coords()]:
        print('ants is loaded: swapping _pseudo_level_order points')
        cb.coord('_pseudo_level_order').points = cb.coord('pseudo_level').points

    da = xr.DataArray.from_iris(cb)
    da.name = 'vegetation_area_fraction'

    # update dim name from ants change
    if '_pseudo_level_order' in [key for key in da.dims]:
        da = da.drop('_pseudo_level_order').swap_dims({'_pseudo_level_order':'pseudo_level'})
    if 'dim0' in [key for key in da.dims]:
        da = da.swap_dims({'dim0':'pseudo_level'})
    if 'dim_0' in [key for key in da.dims]:
        da = da.swap_dims({'dim_0':'pseudo_level'})

    pseudo_map = {  1: 'broad_leaf',
                    2: 'needle_leaf',
                    3: 'c3_grass',
                    4: 'c4_grass',
                    5: 'shrub',
                    6: 'urban',
                    7: 'lake',
                    8: 'soil',
                    9: 'ice',
                    601: 'roof',
                    602: 'canyon'
                }
    
    # convert to dataset with named variables
    ds = xr.Dataset({pseudo_map[i]: da.sel(pseudo_level=i).drop('pseudo_level') for i in da.pseudo_level.values})
    
    # set attributes
    ds.attrs = dict(
                title = title,
                source = fname,
                author = f'{__author__}: {__email__}',
                institution = __institution__
                )
    
    return ds

def combine_tiles(ds):
    '''combines tiles into a single array'''
    ds['trees'] = ds.broad_leaf + ds.needle_leaf
    ds['grass'] = ds.c3_grass + ds.c4_grass
    if 'roof' in list(ds.keys()):
        ds['urban'] = ds.roof + ds.canyon
    return ds

def create_cmap(name,num):
    '''replaces lowest value in colormap with black'''

    from matplotlib.colors import LinearSegmentedColormap

    cmap = plt.get_cmap(name,num)
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # cmaplist[0] = (1.0,1.0,1.0,1.0) # white at zero
    cmaplist[0] = (0.0,0.0,0.0,1.0) # black at zero
    cmap = LinearSegmentedColormap.from_list('mcm',cmaplist, cmap.N)

    return cmap

def distance_bar(ax,distance=10):

    import cartopy.geodesic as cgeo
    # import matplotlib.patheffects as pe

    # stroke = [pe.Stroke(linewidth=1.5, foreground='w'), pe.Normal()]

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    # ydist = abs(ylims[1]-ylims[0])

    xdist = abs(xlims[1]-xlims[0])
    offset = 0.03*xdist

    # plot distance bar
    start = (xlims[0]+offset,ylims[0]+offset)
    end = cgeo.Geodesic().direct(points=start,azimuths=90,distances=distance*1000).flatten()
    ax.plot([start[0],end[0]],[start[1],end[1]], color='k', linewidth=1.5)
    ax.text(start[0]+offset/7,start[1]+offset/5, f'{distance} km', color='black', 
        fontsize=9, ha='left',va='bottom') #  path_effects=stroke

    return ax

def plot_landcover_fractions(ds,title,source,
    plot_order=['broad_leaf', 'needle_leaf', 'c3_grass', 'c4_grass', 'shrub', 'urban', 'lake', 'soil', 'ice'],
    ncols=3,cmap='viridis'):

    import matplotlib as mpl
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    print(f'plotting {title}')

    # find totals
    total = ds.to_array().sum(axis=0,skipna=False)
    land_pts = int(total.count())

    # create custom cmap with black lower, red over
    cmap = create_cmap(cmap,10000)
    cmap.set_over('red')

    stroke = [mpl.patheffects.Stroke(linewidth=1.5, foreground='k'), mpl.patheffects.Normal()]

    # calculate number of rows if three plot_order columns
    nrows = int(np.ceil(len(plot_order)/ncols))
    wdt = 4*ncols
    hgt = (nrows*3)+1
    
    # plot
    plt.close('all')
    fig,axes = plt.subplots(nrows=nrows,ncols=ncols,figsize=(wdt,hgt), sharey=True, sharex=True)

    for ax,key in zip(axes.flatten(), plot_order):

        im = ds[key].plot(ax=ax,vmin=0,vmax=1,cmap=cmap, extend='max',add_colorbar=False)

        # points label
        key_pts = int(ds[key].where(ds[key]>0).count())
        ax.text(0.01,0.99,f'n={key_pts}\n{key_pts/land_pts:.2%}', path_effects=stroke,
            fontsize=6, color='white', ha='left', va='top', transform=ax.transAxes)

        ax.set_ylabel('')
        ax.set_xlabel('')
        ax.set_title(key)

    fig.suptitle(title,x=0.49)
    fig.subplots_adjust(wspace=0.1,hspace=0.15,right=0.85,top=0.92)

    for ax in axes[:,-1].flatten():
        cax = inset_axes(ax,
            width='5%',  # % of parent_bbox width
            height='100%',
            loc='lower left',
            bbox_to_anchor=(1.05, 0, 1, 1),
            bbox_transform=ax.transAxes,
            borderpad=0,
            )
        cbar = mpl.colorbar.ColorbarBase(cax, cmap=im.cmap, norm = im.norm)
        cbar.ax.set_ylabel('landcover cover fraction')
        # cbar.ax.tick_params(labelsize=8)

    ax = distance_bar(ax,distance=100)

    fig.text(0.48,0.06,source,ha='center',va='bottom',fontsize=6)

    return fig

def plot_tile(ds,title,source,key,bounds=(130,155,-44,-24),cmap='viridis'):

    xmin,xmax,ymin,ymax = bounds
    cmap = create_cmap(cmap,1000)

    plt.close('all')
    fig,ax = plt.subplots(figsize=(12,8))
    
    ds[key].sel(latitude=slice(ymin,ymax),longitude=slice(xmin,xmax)).plot(ax=ax,cmap=cmap,vmin=0,vmax=1)
    ax.set_title(f'{title} {key}')
    ax = distance_bar(ax,distance=100)

    fig.text(0.45,0.03,source,ha='center',va='bottom',fontsize=6)

    return fig

