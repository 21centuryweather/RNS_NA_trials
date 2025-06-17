import iris

def make_mp4(fnamein,fnameout,fps=9,quality=26):
    '''
    Uses ffmpeg to create mp4 with custom codec and options for maximum compatability across OS.
        fnamein (string): The image files to create animation from, with wildcards (*).
        fnameout (string): The output filename (excluding extension)
        fps (float): The frames per second.
        quality (float): quality ranges 0 to 51, 51 being worst.
    '''

    import glob
    import imageio.v2 as imageio

    # collect animation frames
    fnames = sorted(glob.glob(fnamein))
    if len(fnames)==0:
        print('no files found to process, check fnamein')
        return
    img_shp = imageio.imread(fnames[0]).shape
    out_h, out_w = img_shp[0],img_shp[1]

    # resize output to blocksize for maximum capatability between different OS
    macro_block_size=16
    if out_h % macro_block_size > 0:
        out_h += macro_block_size - (out_h % macro_block_size)
    if out_w % macro_block_size > 0:
        out_w += macro_block_size - (out_w % macro_block_size)

    # quality ranges 0 to 51, 51 being worst.
    assert 0 <= quality <= 51, "quality must be between 1 and 51 inclusive"

    # use ffmpeg command to create mp4
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{fnamein}" \
        -vcodec libx264 -crf {quality} -s {out_w}x{out_h} -pix_fmt yuv420p -y {fnameout}.mp4'
    os.system(command)

    return f'completed, see: {fnameout}.mp4'

def get_variable_opts(variable):
    '''standard variable options for plotting. to be updated within master script as needed
    
    constraint: iris constraint for the variable used to extract data from the cube
                this can be a long_name, STASH code, or an iris.Constraint() object
    stash: STASH code for the variable (if available)
    plot_title: title of the plot (spaces allowed)
    plot_fname: description used for filename (spaces not allowed)
    units: units of the variable
    obs_key: key used to describe the obs data (if available)
    obs_period: period to resample the obs data to
    fname: filename of the data file to extract from
    vmin: minimum value for the colorbar for variable
    vmax: maximum value for the colorbar for variable
    cmap: colormap to use for the variable
    threshold: threshold error statitics/ benchmarks (if defined)
    fmt: format string for the variable error statistics
    dtype: data type for saving the variable to netcdf
    level: level of the variable (e.g. soil level, 0-indexed)
    '''

    # standard ops
    opts = {
        'constraint': variable,
        'plot_title': variable.replace('_',' '),
        'plot_fname': variable.replace(' ','_'),
        'units'     : '?',
        'obs_key'   : 'None',
        'obs_period': '1H',
        'fname'     : 'umnsaa_pvera',
        'vmin'      : None, 
        'vmax'      : None,
        'cmap'      : 'viridis',
        'threshold' : None,
        'fmt'       : '{:.2f}',
        'dtype'     : 'float32'
        }
    
    if variable == 'air_temperature':
        opts.update({
            'constraint': 'air_temperature',
            'plot_title': 'air temperature (1.5 m)',
            'plot_fname': 'air_temperature_1p5m',
            'units'     : '°C',
            'obs_key'   : 'Tair',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'threshold' : 2,
            'fmt'       : '{:.2f}',
            })
        
    if variable == 'anthropogenic_heat_flux':
        opts.update({
            'constraint': 'm01s03i721',
            'plot_title': 'anthropogenic heat flux',
            'plot_fname': 'anthrop_heat',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : 0,
            'vmax'      : 80,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })
        
    if variable == 'upward_air_velocity':
        opts.update({
            'constraint': 'upward_air_velocity',
            'plot_title': 'upward air velocity',
            'plot_fname': 'upward_air_velocity',
            'units'     : 'm s-1',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : -1,
            'vmax'      : 1,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'updraft_helicity_max':
        opts.update({
            'constraint': 'm01s20i080',
            'plot_title': 'maximum updraft helicity 2000-5000m',
            'plot_fname': 'updraft_helicity_2000_5000m_max',
            'units'     : 'm2 s-2',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pg',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'fmt'       : '{:.1f}',
            })
        
    if variable == 'surface_altitude':
        opts.update({
            'constraint': 'surface_altitude',
            'units'     : 'm',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 2000,
            'cmap'      : 'twilight',
            'dtype'     : 'int16',
            'fmt'       : '{:.0f}',
            })
        
    elif variable == 'dew_point_temperature':
        opts.update({
            'constraint': 'dew_point_temperature',
            'plot_title': 'dew point temperature (1.5 m)',
            'plot_fname': 'dew_point_temperature_1p5m',
            'units'     : '°C',
            'obs_key'   : 'Tdp',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : -10,
            'vmax'      : 30,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'relative_humidity':
        opts.update({
            'constraint': 'relative_humidity',
            'plot_title': 'relative humidity (1.5 m)',
            'plot_fname': 'relative_humidity_1p5m',
            'units'     : '%',
            'obs_key'   : 'RH',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.2f}',
            'dtype'     : 'float32',
            })

    elif variable == 'specific_humidity':
        opts.update({
            'constraint': 'm01s03i237',
            'plot_title': 'specific humidity (1.5 m)',
            'plot_fname': 'specific_humidity_1p5m',
            'units'     : 'kg/kg',
            'obs_key'   : 'Qair',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : 0.004,
            'vmax'      : 0.020,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.4f}',
            })

    elif variable == 'specific_humidity_lowest_atmos_level':
        opts.update({
            'constraint': 'm01s00i010',
            'plot_title': 'specific humidity (lowest atmos. level)',
            'plot_fname': 'specific_humidity_lowest_atmos_level',
            'units'     : 'kg/kg',
            'obs_key'   : 'Qair',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : 0.004,
            'vmax'      : 0.020,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.4f}',
            })

    elif variable == 'evaporation_from_soil_surface':
        opts.update({
            'constraint': 'Evaporation from soil surface',
            'plot_title': 'Evaporation from soil surface',
            'plot_fname': 'Evap_soil',
            'units'     : 'kg/m2/s',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : 0, 
            'vmax'      : 0.0002,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.4f}',
            })

    elif variable == 'latent_heat_flux':
        opts.update({
            'constraint': 'surface_upward_latent_heat_flux',
            'plot_title': 'Latent heat flux',
            'plot_fname': 'latent_heat_flux',
            'units'     : 'W/m2',
            'obs_key'   : 'Qle',
            # 'fname'     : 'umnsaa_pvera',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -100, 
            'vmax'      : 500,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'sensible_heat_flux':
        opts.update({
            'constraint': 'surface_upward_sensible_heat_flux',
            'plot_title': 'Sensible heat flux',
            'plot_fname': 'sensible_heat_flux',
            'units'     : 'W/m2',
            'obs_key'   : 'Qh',
            # 'fname'     : 'umnsaa_pvera',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -100, 
            'vmax'      : 600,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'ground_heat_flux':
        opts.update({
            'constraint': 'm01s03i202',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -100, 
            'vmax'      : 600,
            'cmap'      : 'turbo_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_net_longwave_flux':
        opts.update({
            'constraint': 'surface_net_downward_longwave_flux',
            'plot_title': 'surface net longwave flux',
            'plot_fname': 'surface_net_longwave_flux',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -250,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_net_shortwave_flux':
        opts.update({
            'constraint': 'm01s01i202',
            'plot_title': 'surface net shortwave flux',
            'plot_fname': 'surface_net_shortwave_flux',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : 0,
            'vmax'      : 1000,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_downwelling_shortwave_flux':
        opts.update({
            'constraint': 'surface_downwelling_shortwave_flux_in_air',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : 0,
            'vmax'      : 1000,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'surface_downwelling_longwave_flux':
        opts.update({
            'constraint': 'surface_downwelling_longwave_flux_in_air',
            'units'     : 'W m-2',
            'fname'     : 'umnsaa_psurfa',
            'vmin'      : -250,
            'vmax'      : 450,
            'cmap'      : 'inferno',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'soil_moisture_l1':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 1)',
            'plot_fname': 'soil_moisture_l1',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l1',
            'level'     : 0,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 50,
            'cmap'      : 'turbo_r',
            })

    elif variable == 'soil_moisture_l2':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 2)',
            'plot_fname': 'soil_moisture_l2',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l2',
            'level'     : 1,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            })
        
    elif variable == 'soil_moisture_l3':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 3)',
            'plot_fname': 'soil_moisture_l3',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l3',
            'level'     : 2,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 5,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            })
        
    elif variable == 'soil_moisture_l4':
        opts.update({
            'constraint': 'moisture_content_of_soil_layer',
            'plot_title': 'soil moisture (layer 4)',
            'plot_fname': 'soil_moisture_l4',
            'units'     : 'kg/m2',
            'obs_key'   : 'soil_moisture_l4',
            'level'     : 3,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 5,
            'vmax'      : 100,
            'cmap'      : 'turbo_r',
            })

    elif variable == 'surface_temperature':
        opts.update({
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 70,
            'cmap'      : 'inferno',
            })

    elif variable == 'boundary_layer_thickness':
        opts.update({
            'constraint': 'm01s00i025',
            'plot_title': 'boundary layer thickness',
            'plot_fname': 'boundary_layer_thickness',
            'units'     : '°C',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 3000,
            'cmap'      : 'turbo_r',
            })

    elif variable == 'surface_air_pressure':
        opts.update({
            'units'     : 'Pa',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 88000,
            'vmax'      : 104000,
            'cmap'      : 'viridis',
            })

    elif variable == 'soil_temperature_l1':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (5cm)',
            'plot_fname': 'soil_temperature_l1',
            'units'     : '°C',
            'level'     : 0.05,
            'obs_key'   : 'Tsoil05',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 10,
            'vmax'      : 40,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'soil_temperature_l2':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (22.5cm)',
            'plot_fname': 'soil_temperature_l2',
            'units'     : '°C',
            'level'     : 0.225,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'soil_temperature_l3':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (67.5cm)',
            'plot_fname': 'soil_temperature_l3',
            'units'     : '°C',
            'level'     : 0.675,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'soil_temperature_l4':
        opts.update({
            'constraint': 'soil_temperature',
            'plot_title': 'soil temperature (200cm)',
            'plot_fname': 'soil_temperature_l4',
            'units'     : '°C',
            'level'     : 2,
            'fname'     : 'umnsaa_pverb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'toa_outgoing_shortwave_flux':
        opts.update({
            'constraint': 'm01s01i208',
            'stash'     : 'm01s01i208',
            'plot_title': 'shortwave radiation flux (toa)',
            'plot_fname': 'toa_shortwave',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 50,
            'vmax'      : 600,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'toa_outgoing_shortwave_flux_corrected':
        opts.update({
            'constraint': 'm01s01i205',
            'stash'     : 'm01s01i205',
            'plot_title': 'shortwave radiation flux (toa)',
            'plot_fname': 'toa_shortwave_corrected',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 50,
            'vmax'      : 600,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'toa_outgoing_longwave_flux':
        opts.update({
            'constraint': 'toa_outgoing_longwave_flux',
            'stash'     : 'm01s02i205',
            'plot_title': 'longwave radiation flux (toa)',
            'plot_fname': 'toa_longwave',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 20,
            'vmax'      : 340,
            'cmap'      : 'Greys',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'toa_outgoing_shortwave_radiation_flux':
        opts.update({
            'constraint': 'm01s01i205',
            'plot_title': 'shortwave radiation flux (toa)',
            'plot_fname': 'toa_shortwave',
            'units'     : 'W/m2',
            'fname'     : 'umnsaa_vera',
            'vmin'      : 0, 
            'vmax'      : 1000,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'wind_speed_of_gust':
        opts.update({
            'constraint': 'wind_speed_of_gust',
            'plot_title': 'wind speed of gust',
            'plot_fname': 'wind_gust',
            'units'     : 'm/s',
            'obs_key'   : 'Wind_gust',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 10,
            'vmax'      : 40,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'wind_u':
        opts.update({
            'constraint': 'm01s03i225',
            'plot_title': '10 m wind: U-component',
            'plot_fname': 'wind_u_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })

    elif variable == 'wind_v':
        opts.update({
            'constraint': 'm01s03i226',
            'plot_title': '10 m wind: V-component',
            'plot_fname': 'wind_v_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 25,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })
    
    elif variable == 'wind_speed':
        opts.update({
            'plot_title': '10 m wind speed',
            'plot_fname': 'wind_speed_10m',
            'units'     : 'm/s',
            'obs_key'   : 'wind',
            'vmin'      : 0,
            'vmax'      : 15,
            'cmap'      : 'turbo',
            'threshold' : 2.57,
            'fmt'       : '{:.2f}',
            })

    elif variable == 'ics_soil_albedo':
        opts.update({
            'constraint': 'soil_albedo',
            'plot_title': 'soil albedo (initial conditions)',
            'plot_fname': 'soil_albdo_ics',
            'units'     : '-',
            'fname'     : 'astart',
            'vmin'      : 0,
            'vmax'      : 0.5,
            'cmap'      : 'turbo',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'radar_reflectivity':
        opts.update({
            'constraint': 'radar_reflectivity_due_to_all_hydrometeors_at_1km_altitude',
            'plot_title': 'Radar reflectivity at 1km',
            'plot_fname': 'radar_reflectivity',
            'units'     : 'dBZ',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0.,
            'vmax'      : 25.,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'air_pressure_at_sea_level':
        opts.update({
            'constraint': 'air_pressure_at_sea_level',
            'plot_title': 'air pressure at sea level',
            'plot_fname': 'air_pressure_at_sea_level',
            'units'     : 'Pa',
            'obs_key'   : 'SLP',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 97000,
            'vmax'      : 103000,
            'cmap'      : 'viridis',
            'fmt'       : '{:.1f}',
            })
        
    elif variable == 'fog_area_fraction':
        opts.update({
            'constraint': 'fog_area_fraction',
            'plot_title': 'fog fraction',
            'plot_fname': 'fog_fraction',
            'units'     : '-',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'Greys',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'visibility':
        opts.update({
            'constraint': 'visibility_in_air',
            'plot_title': 'visibility',
            'plot_fname': 'visibility',
            'units'     : 'm',
            'obs_key'   : 'visibility',
            'fname'     : 'umnsaa_pvera',
            'vmin'      : 0,
            'vmax'      : 12000,
            'cmap'      : 'viridis_r',
            'fmt'       : '{:.1f}',
            })

    elif variable == 'cloud_area_fraction':
        opts.update({
            'constraint': 'm01s09i217',
            'units'     : '1',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'Greys_r',
            'fmt'       : '{:.3f}',
            })

    elif variable == 'total_precipitation_rate':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_flux',
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'precipitation rate',
            'plot_fname': 'total_precipitation_rate',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.5f}',
            })
                
    elif variable == 'precipitation_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'precipitation accumulation',
            'plot_fname': 'prcp_accum',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'convective_rainfall_amount_accumulation':
        opts.update({
            'constraint': iris.Constraint(
                name='convective_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'convective rainfall amount accumulation',
            'plot_fname': 'conv_rain_accum',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'convective_rainfall_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s05i201', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'convective rainfall amount',
            'plot_fname': 'convective_rainfall_amount',
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
        
    elif variable == 'convective_rainfall_flux':
        opts.update({
            'constraint': iris.Constraint(
                name='m01s05i205', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'convective rainfall flux',
            'plot_fname': 'convective_rainfall_flux',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.5f}',
            })
        
    elif variable == 'stratiform_rainfall_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'stratiform_rainfall_flux':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_flux', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'units'     : 'kg m-2 s-1',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.5f}',
            })
        
    elif variable == 'daily_precipitation_amount':
        opts.update({
            'constraint': iris.Constraint(
                name='precipitation_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'daily precipitation amount',
            'plot_fname': 'daily_prcp',
            'units'     : 'mm per day',
            'obs_key'   : 'precip_last_aws_obs',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })
    
    elif variable == 'stratiform_rainfall_amount_10min':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_amount', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'rain accumulation',
            'plot_fname': 'rain_accum_10min',
            'units'     : 'kg m-2',
            'obs_key'   : 'None',
            'fname'     : 'umnsaa_*_spec',
            'vmin'      : 0,
            'vmax'      : 200,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'stratiform_rainfall_flux_mean':
        opts.update({
            'constraint': iris.Constraint(
                name='stratiform_rainfall_flux', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'rain flux',
            'plot_fname': 'rain_flux',
            'units'     : r'mm h${^-1}$',
            'obs_key'   : 'precip_hour',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 32,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.6f}',
            })


    elif variable == 'low_type_cloud_area_fraction':
        opts.update({
            'constraint': iris.Constraint(
                name='low_type_cloud_area_fraction', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='mean', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
            'plot_title': 'low cloud fraction',
            'plot_fname': 'low_cloud_fraction',
            'units'     : '0-1',
            'fname'     : 'umnsaa_pverb',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'gist_earth_r',
            'fmt'       : '{:.8f}',
            })

    elif variable == 'surface_runoff_amount':
        opts.update({
            'constraint': 'surface_runoff_amount',
            'plot_title': 'surface runoff amount',
            'plot_fname': 'surface_runoff_amount',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : 0,
            'vmax'      : 100,
            'cmap'      : 'Blues',
            'fmt'       : '{:.1f}',
            })
    

    elif variable == 'subsurface_runoff_amount':
        opts.update({
            'constraint': 'subsurface_runoff_amount',
            'plot_title': 'subsurface runoff amount',
            'plot_fname': 'subsurface_runoff_amount',
            'units'     : 'kg m-2',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : None,
            'vmax'      : None,
            'cmap'      : 'cividis',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'surface_runoff_flux':
        opts.update({
            'constraint': 'surface_runoff_flux',
            'plot_title': 'surface runoff flux',
            'plot_fname': 'surface_runoff_flux',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : 0,
            'vmax'      : 0.01,
            'cmap'      : 'Blues',
            'fmt'       : '{:.4f}',
            })
    
    elif variable == 'subsurface_runoff_flux':
        opts.update({
            'constraint': 'subsurface_runoff_flux',
            'plot_title': 'subsurface runoff flux',
            'plot_fname': 'subsurface_runoff_flux',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfb',
            'vmin'      : None,
            'vmax'      : 0.001,
            'cmap'      : 'cividis',
            'fmt'       : '{:.5f}',
            })

    elif variable == 'surface_total_moisture_flux':
        opts.update({
            'constraint': 'm01s03i223',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : None,
            'vmax'      : 0.0002,
            'cmap'      : 'cividis',
            'fmt'       : '{:.6f}',
            })

    elif variable == 'moisture_flux_u':
        opts.update({
            'constraint': 'm01s30i462',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : None,
            'vmax'      : 0.0002,
            'cmap'      : 'cividis',
            'fmt'       : '{:.6f}',
            })
    
    elif variable == 'moisture_flux_v':
        opts.update({
            'constraint': 'm01s30i463',
            'units'     : 'kg m-2 s-1',
            'fname'     : 'umnsaa_psurfc',
            'vmin'      : None,
            'vmax'      : 0.0002,
            'cmap'      : 'cividis',
            'fmt'       : '{:.6f}',
            })


    elif variable == 'upward_air_velocity_at_300m':
        opts.update({
            'constraint': iris.Constraint(name='upward_air_velocity', height=300.),
            'units'     : 'm s-1',
            'fname'     : 'umnsaa_pb',
            'vmin'      : -2,
            'vmax'      : 2,
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'upward_air_velocity_at_1000m':
        opts.update({
            'constraint': iris.Constraint(name='upward_air_velocity', height=1000.),
            'units'     : 'm s-1',
            'fname'     : 'umnsaa_pb',
            'vmin'      : -2,
            'vmax'      : 2,
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })

        
    elif variable == 'air_temperature_10min':
        opts.update({
            'constraint': 'air_temperature',
            'plot_title': 'air temperature (1.5 m)',
            'plot_fname': 'air_temperature_1p5m',
            'units'     : '°C',
            'obs_key'   : 'Tair',
            'fname'     : 'umnsa_spec',
            'vmin'      : 0,
            'vmax'      : 50,
            'cmap'      : 'inferno',
            'threshold' : 2,
            'fmt'       : '{:.2f}',
            })

    elif variable == 'wind_speed_of_gust_10min':
        opts.update({
            'constraint': iris.Constraint(
                name='wind_speed_of_gust', 
                cube_func=lambda cube: cube.cell_methods == ()),
                'plot_title': 'wind speed of gust',
                'plot_fname': 'wind_gust',
                'units'     : 'm/s',
                'obs_key'   : 'Wind_gust',
                'fname'     : 'umnsa_spec',
                'vmin'      : 10,
                'vmax'      : 40,
                'cmap'      : 'turbo',
                'fmt'       : '{:.2f}',
                })

    elif variable == 'max_wind_speed_of_gust_10min':
        opts.update({
            'constraint': iris.Constraint(
                name='wind_speed_of_gust', 
                cube_func=lambda cube: iris.coords.CellMethod(
                    method='maximum', coords='time', intervals='1 hour'
                    ) in cube.cell_methods),
                'plot_title': 'max wind speed of gust',
                'plot_fname': 'wind_gust_max',
                'units'     : 'm/s',
                'fname'     : 'umnsa_spec',
                'vmin'      : 10,
                'vmax'      : 40,
                'cmap'      : 'turbo',
                'fmt'       : '{:.2f}',
                })

    elif variable == 'moisture_convergence':
        opts.update({
            # 'units'     : 'kg m-2 s-1',
            # 'vmin'      : -0.0004,
            # 'vmax'      : 0.0004,
            'vmin'      : -20,
            'vmax'      : 20,
            'units'     : 'mm/day',
            'cmap'      : 'bwr',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'landfrac':
        opts.update({
            'constraint': 'm01s00i216',
            'plot_title': 'land fraction',
            'plot_fname': 'land_fraction',
            'units'     : '1',
            'fname'     : 'astart',
            'vmin'      : 0,
            'vmax'      : 1,
            'cmap'      : 'viridis',
            'fmt'       : '{:.2f}',
            })

    elif variable == 'orography':
        opts.update({
            'constraint': 'surface_altitude',
            'stash'     : 'm01s00i033',
            'plot_title': 'orography',
            'plot_fname': 'orography',
            'units'     : 'm',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 2500,
            'cmap'      : 'terrain',
            'fmt'       : '{:.0f}',
            })

    elif variable == 'land_sea_mask':
        opts.update({
            'constraint': 'land_binary_mask',
            'stash'     : 'm01s00i030',
            'plot_title': 'land sea mask',
            'plot_fname': 'land_sea_mask',
            'units'     : 'm',
            'fname'     : 'umnsaa_pa000',
            'vmin'      : 0,
            'vmax'      : 1,
            'fmt'       : '{:.1f}',
            'dtype'     : 'int16',
            })

    # add variable to opts
    opts.update({'variable':variable})

    return opts
