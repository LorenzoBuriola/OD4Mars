#This is the main of the pipeline to compute Martian gas Optical Depths
# Lorenzo Buriola - University of Bologna, CNR-ISAC

import argparse
import json
import logging
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from src.logger_setup import setup_logger
from src.generate_profiles import generate_profiles
from src.generate_p_levels import generate_p_levels
from src.generate_mean_profile import generate_mean_profiles, add_altitude, write_mean_cfg
from src.generate_cfg4OD import generate_OD_cfg
from src.generate_OD import generate_OD
from src.OD import OD_calc
from src.OD_fit import OD_fit
from src.input4pack import input4pack, run_packoneband

def parse_args():
    parser = argparse.ArgumentParser(description="My pipeline")
    parser.add_argument(
        "-c", "--config",
        default="settings.json",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level, default is INFO",
    )
    return parser.parse_args()

def main(args):
    # --- Load config file ---
    config = load_config(args.config)

    # Setup logger
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_logger(f'/home/buriola/OD4Mars/NO_BACKUP/log/OD4Mars_{timestamp}.log', getattr(logging, args.log_level.upper(), logging.INFO))
    logger = logging.getLogger(__name__)

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logger.critical(
            "UNCAUGHT EXCEPTION",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    sys.excepthook = handle_exception

    logger.info("OD4Mars program started")
    logger.info(f"Using configuration file: {args.config}")
    logger.info(f"Logging level set to: {args.log_level.upper()}")

    # important path
    defaul_data_path = '/home/buriola/OD4Mars/NO_BACKUP/data/'
    data_path = f"{config.get('data_path', defaul_data_path)}"
    cfg_path = data_path + 'cfg/'
    lyr_path = data_path + 'lyr/'
    lyo_path = data_path + 'lyo/'
    od_path = data_path + 'od/'
    coeff_path = data_path + 'coeff/'
    sMars_path = data_path + 's4Mars/'
    for path in [cfg_path, sMars_path]:
        Path(path).mkdir(parents=True, exist_ok=True)

    flag_profile = config.get('profiles_compute', True)
    flag_p_levels = config.get('pressure_levels_compute', True)
    flag_mean_profile = config.get('mean_profile_compute', True)

    if (flag_profile or flag_p_levels or flag_mean_profile):
        lat_step = 5.625    # from MCD
        long_step = 3.75    # from MCD
        latitudes = np.arange(config.get('profile_latitudes', [-90, 90])[0],
                                config.get('profile_latitudes', [-90, 90])[1]+lat_step, lat_step)
        longitudes = np.arange(config.get('profile_longitudes', [0, 360])[0],
                                 config.get('profile_longitudes', [0, 360])[1], long_step)
        start_date = config.get('profile_dates', ['2019-03-23', '2021-02-07', 24])[0]
        end_date = config.get('profile_dates', ['2019-03-23', '2021-02-07', 24])[1]
        dates = pd.date_range(
            start=start_date,
            end=end_date,
            periods=config.get('profile_dates', ['2019-03-23', '2021-02-07', 24])[2],
            unit='s'
        )
        p_filename = config.get('pressure_levels_file', 'p_edges.npy')

    # Step 1: Generate profiles
    if flag_profile:
        logger.info("Generating profiles")
        generate_profiles(opath = f'{cfg_path}profiles/', dates = dates,
                          latitudes = latitudes,
                          longitudes = longitudes)
    else:
        logger.info("Skipping profile generation")
    logger.info(f"Profiles at '{cfg_path}profiles/'")

    # Step 2: Compute pressure levels
    if flag_p_levels:
        logger.info("Computing pressure levels")
        generate_p_levels(latitudes, longitudes, dates, f'{cfg_path}profiles/',
                          ofile = p_filename)
    else:
        logger.info("Skipping pressure level computation")
    logger.info(f"Pressure levels saved at '{config.get('pressure_levels_file', 'p_edges.npy')}'")

    # Step 3: Compute mean profile
    csv_mean = config.get('mean_profile_file', 'mean_profile.csv')
    if flag_mean_profile:
        logger.info("Computing mean profile")
        df_mean = generate_mean_profiles(latitudes, longitudes, dates, f'{cfg_path}profiles/', p_filename,
                                           csv_ofile = csv_mean)
    else:
        logger.info("Skipping mean profile computation")
        df_mean = pd.read_csv(csv_mean, header=[0,1])['Mean']
    mean_file = f'{cfg_path}{config.get('mean_profile_cfg_file', 'mean_profile.txt')}'
    flag_altitude = config.get('mean_profile_compute_altitude', True)
    if flag_altitude:
        add_altitude(df_mean)
    else:
        df_mean.drop(columns=['Altitude'], errors='ignore')
    write_mean_cfg(df_prof=df_mean, ofile=mean_file)
    logger.info(f"Mean profile saved at '{cfg_path}{config.get('mean_profile_cfg_file', 'mean_profile.txt')}'")

    flag_od = config.get('od_compute', True)
    flag_bin = config.get('od_bin', True)
    flag_fit = config.get('od_fit', True)
    flag_sMars = config.get('for_sMars', True)

    if (flag_od or flag_bin or flag_fit or flag_sMars):
        gas_list = config.get('gas_list', ["CO2", "CO", "H2O", "O3", "HCl", "HDO"])
        for path in [lyo_path, od_path, coeff_path, lyr_path]:
            for g_name in gas_list:
                Path(f'{path}/{g_name}/').mkdir(parents=True, exist_ok=True)
            Path(path).mkdir(parents=True, exist_ok=True)
        ranges = np.arange(config.get('ranges', [90, 3010, 40])[0],
                            config.get('ranges', [90, 3010, 40])[1]+config.get('ranges', [90, 3010, 40])[2],
                            config.get('ranges', [90, 3010, 40])[2])
        temperatures = np.arange(config.get('temperatures', [-60, 60, 10])[0],
                            config.get('temperatures', [-60, 60, 10])[1]+config.get('temperatures', [-60, 60, 10])[2],
                            config.get('temperatures', [-60, 60, 10])[2])
        ranges_bin = ranges = np.arange(config.get('od_bin_ranges', [90, 3010, 40])[0],
                            config.get('od_bin_ranges', [90, 3010, 40])[1]+config.get('od_bin_ranges', [90, 3010, 40])[2],
                            config.get('od_bin_ranges', [90, 3010, 40])[2])
        degree = config.get('fit_degree', 3)
    
    if flag_od:
        # Step 4: Generate cfg file for each species
        logger.info("Generating cfg files for OD computation")
        
        generate_OD_cfg(gas_list, cfg_path+'mean_profile.txt', f'{cfg_path}OD_gen/')
        logger.info(f"OD cfg files saved at '{cfg_path}OD_gen/'")

        # Step 5: Generate OD
        logger.info("Generating Optical Depths")
        generate_OD(gas_list, ranges-0.005, temperatures, cfg_path, lyo_path, lyr_path)
    else:
        logger.info("Skipping Optical Depth Generation")
    logger.info(f'OD at high resolution stored ar {lyo_path}')
    low_res = config.get('od_low_res', 1e-2)
    logger.info(f'high resolution: 1e-4 cm-1, low resolution: {low_res:.0e} cm-1')

    if flag_bin:
        # Step 6: Binning OD
        cumulative = config.get('od_bin_cumulative', 'layer')
        if cumulative not in ['top', 'bottom', 'layer']:
            cumulative = 'layer'
            logger.warning(f"Invalid cumulative value '{cumulative}' provided. Defaulting to 'layer'.")
        if low_res <= 1e-4:
            cumulative = 'layer'
            logger.warning(f'OD fitted at the higher resolution, no binning is performed')
        logger.info(f'Binning OD with cumulative method: {cumulative}')
        OD_calc(gas_list, ranges_bin-0.005, temperatures, lyo_path, od_path, low_res, cumulative)
    logger.info(f'OD stored at {od_path}')

    if flag_fit:
        logger.info('Fitting OD')
        logger.info(f'Fitting with degree {degree} and low resolution {low_res:.0e} cm-1')
        OD_fit(gas_list, ranges_bin, 
               degree,
               od_path, coeff_path, low_res, cumulative)
    else:
        logger.info('Skipping OD fitting')
    logger.info(f'Fit stored at {coeff_path}')

    if flag_sMars:
        logger.info('Preparing input for s4Mars')
        logger.info(f'Using fit with degree {degree} and low resolution {low_res:.0e} cm-1')
        name_database = config.get('name_od_database', 'PSG')
        outdir = Path(sMars_path+name_database+'/to_pack/')
        outdir.mkdir(parents=True, exist_ok=True)
        input4pack(gas_list,
                   ranges_bin,
                   degree,
                   coeff_path,
                   outdir, low_res, cumulative)
        logger.info(f'run packoneband fortran executable')
        run_packoneband('/home/buriola/OD4Mars/src/.', name_database, str(degree), cumulative, outfile=config.get('pack_oneband_file', 'pack_oneband_output.txt'))
    else:
        logger.info("Skipping input for s4Mars preparation")

    logger.info('Program OD4Mars executed!')

def load_config(path):
    """Load configuration settings from a JSON file."""
    with open(path, "r") as f:
        config = json.load(f)
    return config

if __name__ == "__main__":
    args = parse_args()
    main(args)