# Lorenzo Buriola - University of Bologna, CNR-ISAC
# The function will create a profile for each combination of latitude, longitude, and date
# in the specified output path.
# 1 in pipeline

import logging
import numpy as np
import pandas as pd
import PSGpy.cfg as cfg
from PSGpy.run_psg import run_psg
from PSGpy.utils import name_file

logger = logging.getLogger(__name__)

def generate_profiles(opath, dates, latitudes, longitudes):
    # INPUTS
    
    dates_list = dates.strftime('%Y/%m/%d %H:%M').to_list()

    # Generate profiles
    logger.info("Generating profiles...")
    cfg_df = {
        'OBJECT' : 'Planet',
        'OBJECT-NAME' : 'Mars',
        'GEOMETRY-REF' : 'User',
        'GEOMETRY' : 'Nadir', 
        'GEOMETRY-OBS-ALTITUDE' : '400.0',
        'GEOMETRY-ALTITUDE-UNIT' : 'km',
        'ATMOSPHERE-STRUCTURE' : 'Model_MCD',
        'GENERATOR-INSTRUMENT':'user',
        'GENERATOR-RANGE1':'100',
        'GENERATOR-RANGE2':'3000',
        'GENERATOR-RANGEUNIT':'cm',
        'GENERATOR-RESOLUTION':'5',
        'GENERATOR-RESOLUTIONUNIT':'cm'
    }

    for date in dates_list:
        for lat in latitudes:
            longs_to_use = [0] if abs(lat) == 90 else longitudes
            for long in longs_to_use:
                logger.info(f"Generating profile for date: {date}, lat: {lat}, long: {long}")
                cfg_df['OBJECT-DATE'] = date
                cfg_df['OBJECT-OBS-LATITUDE'] = str(lat)
                cfg_df['OBJECT-OBS-LONGITUDE'] = str(long)
                cfg.dict_to_cfg(cfg_dict=cfg_df, file_path='cfg_temp.cfg')
                run_psg(cfg_file='cfg_temp.cfg', kind='cfg', wephm = 'y', watm='y',
                    out_file=f"{opath}{name_file('cfg', date, lat, long)}.cfg", verbose=False)
    logger.info("Profile generation completed.")

