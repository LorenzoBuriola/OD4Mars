#This is the main of the pipeline to compute Martian gas Optical Depths
# Lorenzo Buriola - University of Bologna, CNR-ISAC

import argparse
import tomllib
import logging
import sys
import time
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

def load_config(path):
    """Load configuration settings from a JSON file."""
    with open(path, "rb") as file:
        config = tomllib.load(file)
    return config

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
    data_path = config['data_path']
    cfg_path = data_path + 'cfg/'
    lyr_path = data_path + 'lyr/'
    lyo_path = data_path + 'lyo/'
    od_path = data_path + 'od/'
    coeff_path = data_path + 'coeff/'
    s4Mars_path = data_path + 's4Mars/'
    for path in [cfg_path, lyr_path, lyo_path, od_path, coeff_path, s4Mars_path]:
        Path(path).mkdir(parents=True, exist_ok=True)

    flag_profile = config['profiles']['flag']
    flag_p_levels = config['pressure_levels']['flag']
    flag_mean_profile = config['mean_profile']['flag']

    if (flag_profile or flag_p_levels or flag_mean_profile):
        lat_step = 5.625    # from MCD
        long_step = 3.75    # from MCD
        latitudes_cfg = config['profiles']['latitudes']
        start, end = latitudes_cfg
        latitudes = np.arange(start, end+lat_step/2, lat_step)
        longitudes_cfg = config['profiles']['longitudes']
        start, end = longitudes_cfg
        longitudes = np.arange(start, end, long_step)
        dates_cfg = config['profiles']['dates']
        start, end, periods = dates_cfg
        dates = pd.date_range(
            start=start,
            end=end,
            periods=periods,
            unit='s'
        )
        p_filename = config['profiles']['path']

    # Step 1: Generate profiles
    if flag_profile:
        logger.info("Generating profiles")
        profile_path = cfg_path + 'profiles/'
        Path(profile_path).mkdir(exist_ok=True)
        generate_profiles(opath = profile_path, dates = dates,
                          latitudes = latitudes,
                          longitudes = longitudes)
    else:
        logger.info("Skipping profile generation")
    logger.info(f"Profiles at '{cfg_path}profiles/'")

    # Step 2: Compute pressure levels
    if flag_p_levels:
        logger.info("Computing pressure levels")
        generate_p_levels(latitudes, longitudes, dates, f'profilepath',
                          ofile = p_filename)
    else:
        logger.info("Skipping pressure level computation")
    logger.info(f"Pressure levels saved at '{config['pressure_levels']['path']}'")

    # Step 3: Compute mean profile
    csv_mean = config['mean_profile']['path_csv']
    if flag_mean_profile:
        logger.info("Computing mean profile")
        df_mean = generate_mean_profiles(latitudes, longitudes, dates, f'{cfg_path}profiles/', p_filename, csv_ofile = csv_mean)
    else:
        logger.info("Skipping mean profile computation")
        df_mean = pd.read_csv(csv_mean, header=[0,1])['Mean']
    mean_file = f"{cfg_path}{config['mean_profile']['path_cfg']}"
    flag_altitude = config['mean_profile']['compute_altitude']
    if flag_altitude:
        add_altitude(df_mean)
    else:
        df_mean = df_mean.drop(columns=['Altitude'], errors='ignore')
    write_mean_cfg(df_prof=df_mean, ofile=mean_file)
    logger.info(f"Mean profile saved at '{mean_file}'")

    flag_od = config['od-lyo']['flag']
    flag_bin = config['od-bin']['flag']
    flag_fit = config['od-fit']['flag']
    flag_s4Mars = config['s4Mars']['flag']
    gas_list = config['gas_list']
    temperatures_cfg = config['temperatures']
    start, end, step = temperatures_cfg
    temperatures = np.arange(start, end+step/2, step)

    if flag_od:
        high_res = config['od-lyo']['high_res']
        logger.info(f'high resolution: {high_res:.0e} cm-1')

        ranges_cfg = config['od-lyo']['ranges']
        start, end, step = ranges_cfg
        ranges = np.arange(start, end+step/2, step)

        # Step 4: Generate cfg file for each species
        logger.info("Generating cfg files for OD computation")
        Path(f'{cfg_path}OD_gen/').mkdir(exist_ok=True)
        t0 = time.time()
        generate_OD_cfg(gas_list, mean_file, f'{cfg_path}OD_gen/')
        logger.info(f"OD generation took {(time.time()-t0)/3600:.2f} h")
        logger.info(f"OD cfg files saved at '{cfg_path}OD_gen/'")

        # Step 5: Generate OD
        for g_name in gas_list:
            Path(f'{lyo_path}{g_name}/').mkdir(parents=True, exist_ok=True)
        logger.info("Generating Optical Depths")
        generate_OD(gas_list, ranges-0.005, high_res, temperatures, cfg_path, lyo_path, lyr_path)
    else:
        logger.info("Skipping Optical Depth Generation")
    logger.info(f'OD at high resolution stored ar {lyo_path}')

    if (flag_bin or flag_fit or flag_s4Mars):
        ranges_bin_cfg = config['od-bin']['ranges']
        start, end, step = ranges_bin_cfg
        ranges_bin = np.arange(start, end+step/2, step)
        low_res = config['od-bin']['low_res']
        logger.info(f'low resolution: {low_res:.0e} cm-1')
        cumulative = config['od-bin']['cumulative']
        if cumulative not in ['top', 'bottom', 'layer', 'od', '']:
            cumulative = 'layer'
            logger.warning(f"Invalid cumulative value '{cumulative}' provided. Defaulting to 'layer'.")
        degree = config['od-fit']['degree']
    
    if flag_bin:
        # Step 6: Binning OD
        if low_res <= 1e-4:
            cumulative = ''
            logger.info('OD fitted at the higher resolution, no binning is performed')
        else: 
            logger.info(f'OD binning from 1e-4 to {low_res:.0e} cm-1')
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

    if flag_s4Mars:
        logger.info('Preparing input for s4Mars')
        logger.info(f'Using fit with degree {degree} and low resolution {low_res:.0e} cm-1')
        name_database = config['s4Mars']['name_od_database']
        outdir = Path(s4Mars_path+name_database+'/to_pack/')
        outdir.mkdir(parents=True, exist_ok=True)
        input4pack(gas_list,
                   ranges_bin,
                   degree,
                   coeff_path,
                   outdir, low_res, cumulative)
        logger.info(f'run packoneband fortran executable')
        run_packoneband('/home/buriola/OD4Mars/src/.', name_database, str(degree), cumulative, outfile=config['s4Mars']['pack_oneband_out'])
    else:
        logger.info("Skipping input for s4Mars preparation")

    logger.info('Program OD4Mars executed!')

if __name__ == "__main__":
    args = parse_args()
    main(args)