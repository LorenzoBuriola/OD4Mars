import logging
import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt

import PSGpy.cfg as cfg
import PSGpy.atm_obj as atm
from PSGpy.utils import name_file

logger = logging.getLogger(__name__)

def generate_mean_profiles(latitudes, longitudes, dates, ipath, p_filename, csv_ofile):
    dates_list = dates.strftime('%Y/%m/%d %H:%M').to_list()
    tt = []
    h2o = []
    co = []
    co2 = []
    o3 = []

    logger.info(f'Reading pressure edges from file {p_filename}...')
    p_edges = np.load(p_filename)
    logger.debug(f'Number of pressure edges: {len(p_edges)}')

    logger.info('Reading profiles from cfg files and interpolating to pressure edges...')
    for date in dates_list:
        for lat in latitudes:
            longs_to_use = [0] if abs(lat) == 90 else longitudes
            for long in longs_to_use:
                temp_cfg = cfg.read_cfg(f"{ipath}{name_file('cfg', date, lat, long)}.cfg")
                temp_df = cfg.read_atm_layers(temp_cfg)[::-1]  # Reverse the order to match pressure edges
                P = temp_df.Pressure.to_numpy()
                T = temp_df.Temperature.to_numpy()
                Water = temp_df.H2O.to_numpy()
                Monoxide = temp_df.CO.to_numpy()
                Dioxide = temp_df.CO2.to_numpy()
                Ozone = temp_df.O3.to_numpy()
                tt.append(np.interp(p_edges,P,T, right=np.nan))
                h2o.append(np.interp(p_edges,P,Water, right=np.nan))
                co.append(np.interp(p_edges,P,Monoxide, right=np.nan))
                co2.append(np.interp(p_edges,P,Dioxide, right=np.nan))
                o3.append(np.interp(p_edges,P,Ozone, right=np.nan))
    
    logger.info('Calculating mean and standard deviation for each pressure level...')
    meanT = np.nanmean(tt, axis=0)[::-1]
    meanCO2 = np.nanmean(co2, axis=0)[::-1]
    meanCO = np.nanmean(co, axis=0)[::-1]
    meanH2O = np.nanmean(h2o, axis=0)[::-1]
    meanO3 = np.nanmean(o3, axis=0)[::-1]

    stdT = np.nanstd(tt, axis=0)[::-1]
    stdCO2 = np.nanstd(co2, axis=0)[::-1]
    stdCO = np.nanstd(co, axis=0)[::-1]
    stdH2O = np.nanstd(h2o, axis=0)[::-1]
    stdO3 = np.nanstd(o3, axis=0)[::-1]

    p_edges = p_edges[::-1]  # Reverse the order to compute altitude correctly

    df_mean = pd.DataFrame({
        'Pressure' : p_edges,
        'Temperature' : meanT,
        'CO2' : meanCO2,
        'CO' : meanCO,
        'H2O' : meanH2O,
        'O3' : meanO3
    })
    df_mean['HCl'] = 1e-9

    df_std = pd.DataFrame({
        'Pressure' : p_edges,
        'Temperature' : stdT,
        'CO2' : stdCO2,
        'CO' : stdCO,
        'H2O' : stdH2O,
        'O3' : stdO3
    })

    logger.info(f'Saving mean profile and standard deviation to {csv_ofile}...')
    combined_df = pd.concat({'Mean': df_mean, 'StdError': df_std},axis=1)
    # Save to CSV
    combined_df.to_csv(csv_ofile, index=False, float_format='%.4e')
    return df_mean

def add_altitude(df):
    logger.info('Computing altitude profile from pressure edges and mean temperature...')
    p_edges = df['Pressure']
    meanT = df['Temperature']
    W_air = 43.61   #Weight [g/mol]
    R = 8314.4598  #Universal gas constant [mJ/(mol*K)]
    g_surf = 3.73     #Gravity [m/s^2]
    R_surf = 6779.8/2*1000  #Mars radius [m]
    hh = np.zeros_like(p_edges)
    logger.debug(f'Altitude at level 1: {hh[0]/1000} km')
    for i in range(len(p_edges)-1):
        g = g_surf*((hh[i]+R_surf)/ R_surf)**2
        H = R * (meanT[i]+meanT[i+1])/2 / (W_air * g)  #Scale height [m]
        hh[i+1] = hh[i] + H * np.log(p_edges[i] / p_edges[i+1])  #Altitude [m]
        logger.debug(f'Altitude at level {i+2}: {hh[i+1]/1000} km')
    df.insert(2, 'Altitude', hh / 1000)

def write_mean_cfg(df_prof, ofile) -> None:
    if isinstance(df_prof, str):
        atm = pd.read_csv(df_prof, header=0)
    else:
        atm = df_prof
    cfg_df = {}
    cfg_df['OBJECT'] = 'Planet'
    cfg_df['OBJECT-NAME'] = 'Mars'
    cfg_df['OBJECT-DATE'] = '2019/03/24 00:00'
    cfg_df['OBJECT-OBS-LATITUDE'] = '0.0'
    cfg_df['OBJECT-OBS-LONGITUDE'] = '90.0'
    cfg_df['OBJECT-GRAVITY'] = '3.933'
    cfg_df['OBJECT-GRAVITY-UNIT'] = 'rho'
    cfg_df['OBJECT-DIAMETER'] = '6779.80'
    cfg_df['GEOMETRY-REF'] = 'User'
    cfg_df['GEOMETRY'] = 'Nadir'
    cfg_df['GEOMETRY-OBS-ALTITUDE'] = '400.0'
    cfg_df['GEOMETRY-OBS-ANGLE'] = '0.0'
    cfg_df['GEOMETRY-ALTITUDE-UNIT'] = 'km'
    cfg_df['GENERATOR-INSTRUMENT'] = 'user'
    cfg_df['GENERATOR-GAS-MODEL'] = 'Y'
    cfg_df['GENERATOR-CONT-MODEL'] = 'N'
    cfg_df['GENERATOR-CONT-STELLAR'] = 'N'
    cfg_df['GENERATOR-RANGEUNIT'] = 'cm'
    cfg_df['GENERATOR-RESOLUTIONUNIT'] = 'cm'
    cfg_df['GENERATOR-RESOLUTIONKERNEL'] = 'N'
    cfg_df['GENERATOR-RADUNITS'] = 'Wsrm2cm'
    cfg_df['GENERATOR-NOISE'] = 'NO'
    cfg_df['ATMOSPHERE-STRUCTURE'] = 'Equilibrium'
    cfg_df['ATMOSPHERE-PUNIT'] = 'bar'
    cfg_df['SURFACE-MODEL'] = 'Lambert'
    cfg_df['SURFACE-PHASEMODEL'] = 'ISO'
    cfg_df['SURFACE-ALBEDO'] = '0.0'
    cfg_df['SURFACE-EMISSIVITY'] = '1.0'
    cfg_df['SURFACE-NSURF'] = '0'
    cfg_df['SURFACE-SURF'] = 'MARS'
    cfg_df['SURFACE-TYPE'] = 'ALBEDO_GSFC'
    cfg_df['SURFACE-ABUN'] = '0'
    cfg_df['SURFACE-UNIT'] = 'pct'
    cfg_df['SURFACE-THICK'] = '1'
    cfg_df['SURFACE-MIXING'] = 'Normalized'
    n_layer = atm.shape[0]
    cfg_df['ATMOSPHERE-WEIGHT'] = '43.64'  # Weight of air [g/mol]
    cfg_df['ATMOSPHERE-LAYERS'] = n_layer - 1  # Number of layers (excluding the surface layer)
    names = atm.columns
    cfg_df['ATMOSPHERE-LAYERS-MOLECULES'] = ",".join(names[2:])
    for i in range(n_layer-1):
        cfg_df[f'ATMOSPHERE-LAYER-{i+1}'] = ','.join('{:.3e}'.format(n) for n in atm.iloc[i+1,:].to_list())
    cfg_df['ATMOSPHERE-PRESSURE'] = atm.loc[0,'Pressure']
    cfg_df['SURFACE-TEMPERATURE'] = atm.loc[0,'Temperature']
    cfg.dict_to_cfg(cfg_df, ofile)
