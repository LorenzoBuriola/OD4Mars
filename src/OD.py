import logging
import numpy as np
import xarray as xr
from PSGpy.utils import read_out
from scipy.stats import binned_statistic

logger = logging.getLogger(__name__)

def OD_calc(gas_list, ranges, temperatures, lyo_path, od_path, low_res = 1e-2):
    tab = read_out(f'{lyo_path}CO2/lyo_CO2_0_freq90_130.txt')
    hh = tab.columns[1:-1].to_numpy(dtype='float64')
    for g_name in gas_list:
        logger.info(f'Gas: {g_name}')
        for i in range(len(ranges)-1):
            logger.info(f'Frequency window: {ranges[i]}-{ranges[i+1]}')
            EE = False
            #low_freqs = np.arange(ranges[i], ranges[i+1], 1e-2) + 0.005
            list_of_od = []
            list_of_mask = []
            for DT in temperatures:
                logger.info(f'Temperature shift: {DT}')
                try:
                    tab = read_out(f'{lyo_path}{g_name}/lyo_{g_name}_{DT}_freq{ranges[i]:.0f}_{ranges[i+1]:.0f}.txt')
                    tab = tab[:400000] # Limit to the first 400000 rows
                    tab = OD_compute(tab)
                    if low_res <= 1e-4:
                        low_res = 1e-4
                        logger.debug(f'Low resolution: {low_res:.0e}, no binning applied')
                        low_freqs = tab.freq.to_numpy()
                        tab=tab.iloc[:,1:]
                        mask = np.ones_like(tab)
                    else:
                        logger.debug(f'Low resolution: {low_res:.0e}, applying binning')
                        tab, mask, low_freqs = OD_binning(tab, 40/low_res)
                    list_of_od.append(tab)
                    list_of_mask.append(mask)
                except ValueError as ERROR:
                    logger.info("An exception occurred:", type(ERROR).__name__, "–", ERROR)
                    EE = True
            if EE:
                continue
            aa = np.stack(list_of_od, axis=-1)
            mm = np.stack(list_of_mask, axis=-1)

            aa = xr.DataArray(data=aa, dims=['freq', 'altitude', 'DeltaT'], coords=dict(
                    freq = low_freqs,
                    altitude = hh,
                    DeltaT = temperatures
                ))
            mm = xr.DataArray(data=mm, dims=['freq', 'altitude', 'DeltaT'], coords=dict(
                    freq = low_freqs,
                    altitude = hh,
                    DeltaT = temperatures
                ))
            ds = xr.Dataset({
                'od': aa,
                'mask': mm
            })
            path = f'{od_path}{g_name}/od_{g_name}_freq{ranges[i]+0.005:.0f}_{ranges[i+1]+0.005:.0f}_{low_res:.0e}.nc'
            ds.to_netcdf(path, engine='netcdf4')

    logger.info('All done!')
 
def OD_compute(data):
    altitude = data.columns[1:].to_numpy(dtype='float64')
    paths = np.diff(altitude).reshape(1,-1)
    df_out = data.iloc[:,:-1].copy()
    df_out.iloc[:,1:] = data.iloc[:,1:-1]*paths
    names = [f'level_{i+1}' for i in range(len(df_out.columns)-1)]
    names.insert(0,'freq')
    df_out.columns = names
    return df_out
    
def OD_binning(high_res, n_bins):
    #Sort values by frequency not to mess up the binning
    high_res.sort_values(by='freq',inplace=True)
    #Get the high frequency and optical depth values
    f_high = high_res.freq.to_numpy()
    ods = high_res.to_numpy()[:,1:].T
    #Compute transmittance and cumulative transmittance
    trn = np.exp(-ods)
    cum_trn = np.cumprod(trn[::-1,:], axis=0)[::-1,:]
    cum_binned,edges,_ = binned_statistic(x=f_high,values=cum_trn,statistic='mean',bins=n_bins)
    binned,_,_ = binned_statistic(x=f_high,values=trn,statistic='mean',bins=n_bins)
    
    binned = np.clip(binned, 1e-300, 1.0)
    od_bin = np.log(binned)

    mask = (cum_binned[:-1] != 0) & (cum_binned[1:] != 0)
    np.divide(cum_binned[:-1], cum_binned[1:], out=od_bin[:-1], where=mask)
    np.log(od_bin[:-1], out=od_bin[:-1], where=mask)
    od_bin = -od_bin

    new_row = np.ones((1, mask.shape[1]), dtype=bool)
    mask = np.vstack((mask, new_row))

    low_freqs = np.round((edges[:-1] + edges[1:]) / 2, 4)
    return od_bin.T, mask.T, low_freqs

"""
    #Compute the error
    NN = np.sqrt(f_high.size/n_bins)
    pr_error,_,_ = binned_statistic(x=f_high,values=trn,statistic='std',bins=n_bins)
    pr_error = pr_error/NN
    #sec_error,_,_ = binned_statistic(x=f_high,values=ods,statistic='std',bins=n_bins)
    #error = sec_error
    error = pr_error/binned



    term1 = np.empty_like(error[:-1])
    term2 = np.empty_like(error[:-1])
    np.divide(pr_error[:-1], binned[:-1], out=term1, where=mask)
    np.square(term1, out=term1)
    np.divide(pr_error[1:], binned[1:], out=term2, where=mask)
    np.square(term2, out=term2)
    np.sqrt(term1 + term2, out=error[:-1], where=mask)
    error[-1] = pr_error[-1]/binned[-1]
"""