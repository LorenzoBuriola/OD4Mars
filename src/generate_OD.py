import logging
import numpy as np
import PSGpy.cfg as cfg
from PSGpy.run_psg import run_psg
import os

logger = logging.getLogger(__name__)
            
def T_shift(DT, cfg_df):
    cfg_out = cfg_df.copy()
    df = cfg.read_atm_layers(cfg_df)
    df['Temperature'] = (df['Temperature']+DT).round(1)
    cfg.write_atm_layers(df,cfg_out)
    cfg_out['SURFACE-TEMPERATURE'] = float(cfg_df['SURFACE-TEMPERATURE']) + DT
    return cfg_out

def generate_OD(gas_list, ranges, temperatures, cfg_path, lyo_path, lyr_path):

    ranges = np.arange(89.995, 3049.995, 40)
    DTs = np.arange(-60, 70, 10)

    logger.info('Starting computing ODs\n')

    for g_name in gas_list:
        logger.info(f'Gas: {g_name}')
        cfg_dict = cfg.read_cfg(f'{cfg_path}OD_gen/cfg_{g_name}.txt')
        for DT in DTs:
            logger.info(f'Temperature shift: {DT}')
            temp = T_shift(DT, cfg_dict)
            for i in range(len(ranges)-1):
                logger.info(f'freqs: {ranges[i]}-{ranges[i+1]}')
                #pp = f"{lyo_path}{g_name}/lyo_{g_name}_{DT}_freq{ranges[i]:.0f}_{ranges[i+1]:.0f}.txt"
                #if os.path.exists(pp):
                #    continue
                temp['GENERATOR-RANGE1'] = "{:.4f}".format(ranges[i])
                temp['GENERATOR-RANGE2'] = "{:.4f}".format(ranges[i+1])
                cfg.dict_to_cfg(temp, f'{cfg_path}OD_gen/cfg_temp.txt')
                run_psg(cfg_file=f'{cfg_path}OD_gen/cfg_temp.txt',
                            kind='lyo',
                            wephm='y',
                            out_file=f"{lyo_path}{g_name}/lyo_{g_name}_{DT}_freq{ranges[i]:.0f}_{ranges[i+1]:.0f}.txt",
                            verbose=False)
                run_psg(cfg_file=f'{cfg_path}OD_gen/cfg_temp.txt',
                            kind='lyr',
                            wephm='y',
                            out_file=f"{lyr_path}{g_name}/lyr_{g_name}_{DT}_freq{ranges[i]:.0f}_{ranges[i+1]:.0f}.txt",
                            verbose=False)
    logger.info("OD generation completed")