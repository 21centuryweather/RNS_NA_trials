## Roughness experiment

Comparing standard (CCIv2) with modified (CCIv2_mod) over 3 month summer period: Dec 2019 - Feb 2020

The only difference between them is that CCIv2_mod has c3/c4 grass roughness increased to match tree roughness:

**GAL9**
```
diff rns_ostia_NA/app/um/opt/rose-app-GAL9.conf rns_ostia_NA/app/um/opt/rose-app-expt1.conf

< z0v_io=1.1,1.1,0.22,0.22,1.0
---
> z0v_io=1.1,1.1,1.1,1.1,1.0
```

**RAL3P2**
```
diff rns_ostia_NA/app/um/opt/rose-app-ral3p2.conf rns_ostia_NA/app/um/opt/rose-app-expt2.conf

< z0v_io=1.1,1.1,0.1,0.1,0.4
---
> z0v_io=1.1,1.1,1.1,1.1,0.4

```

[Code for producing figures](plotting/plot_pp.py)  
[Experiment MOSRS repository](https://code.metoffice.gov.uk/trac/roses-u/browser/d/g/7/6/8/rns_ostia_NA)

## Domains

![NA domains](ancils/figures/NA_domains_surface_altitude.png)

## Precipitation per day

**GAL9** (total_precipitation)
![figures/total_precipitation_rate_diff_GAL9.png](figures/total_precipitation_rate_diff_GAL9.png)

**RAL3p2** (stratiform_rainfall_flux)
![stratiform_rainfall_flux_diff_RAL3P2.png](figures/stratiform_rainfall_flux_diff_RAL3P2.png)

## Moisture convergence

**GAL9**
![moisture_convergence_diff_GAL9.png](figures/moisture_convergence_diff_GAL9.png)

**RAL3p2**
![moisture_convergence_diff_RAL3P2.png](figures/moisture_convergence_diff_RAL3P2.png)

**RAL3p2 coarsened**
![moisture_convergence_diff_RAL3P2_coarsened.png](figures/moisture_convergence_diff_RAL3P2_coarsened.png)

## Wind speed at 10 m

**RAL3p2**
![wind_speed_10m_diff_RAL3P2.png](figures/wind_speed_10m_diff_RAL3P2.png)

## Upward air velocity at 300 m

**RAL3p2 coarsened** 
![upward_air_velocity_at_300m_diff_RAL3P2_coarsened.png](figures/upward_air_velocity_at_300m_diff_RAL3P2_coarsened.png)

## Upward air velocity at 1000 m

**RAL3p2 coarsened** 
![upward_air_velocity_at_1000m_diff_RAL3P2_coarsened.png](figures/upward_air_velocity_at_1000m_diff_RAL3P2_coarsened.png)


