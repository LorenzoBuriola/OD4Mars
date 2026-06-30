import logging
import numpy as np
import pandas as pd
import PSGpy.cfg as cfg
from PSGpy.utils import name_file

logger = logging.getLogger(__name__)

def custom_edges(aa, nbin):
    nn = len(aa)
    return np.interp(np.linspace(0, nn, nbin + 1),
                     np.arange(nn),
                     np.sort(aa))

def generate_p_levels(latitudes, longitudes, dates, p_filename, ofile):
    logger.info("Generating pressure levels...")
    dates_list = dates.strftime('%Y/%m/%d %H:%M').to_list()
    p = []
    for date in dates_list:
        for lat in latitudes:
            longs_to_use = [0] if abs(lat) == 90 else longitudes
            for long in longs_to_use:
                temp_cfg = cfg.read_cfg(f"{p_filename}{name_file('cfg', date, lat, long)}.cfg")
                temp_df = cfg.read_atm_layers(temp_cfg)
                p.append(temp_df.Pressure)
    p = np.asarray(p)
    pcut1 = 1e-8
    pcut2 = 1e-3
    plow = p[p <= pcut1]
    e_low = custom_edges(plow, 5)
    p_med = p[(p>pcut1) & (p<pcut2)]
    e_med = custom_edges(p_med, 20)[1:-1]  # Exclude the last edge to avoid duplication
    p_high = p[p>=pcut2]
    e_high = custom_edges(p_high,30)
    ee = np.unique(np.concatenate([e_low,e_med,e_high]))
    
    logger.debug('******************************')
    logger.debug(f"Pressure levels generated:")
    for i in range(len(ee)):
        logger.debug(f"Level {i+1}:\t{ee[i]:.3}")

    np.save(ofile, ee)
    logger.info("Pressure level generation completed.")

    return ee
