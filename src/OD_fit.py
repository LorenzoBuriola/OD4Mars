import logging
import xarray as xr
import numpy as np

logger = logging.getLogger(__name__)

def weighted_polyfit(y, sigma, T, degree):
    """Fit a weighted polynomial of given degree to data y(T) with error sigma."""
    if np.any(~np.isfinite(y)) or np.any(~np.isfinite(sigma)):
        return np.full(degree + 1, np.nan)
    try:
        w = 1 / sigma
        p = np.polyfit(T, y, deg=degree, w=w)
        return p  # returns [a3, a2, a1, a0]
    except Exception:
        return np.full(degree + 1, np.nan)
    
def OD_fit(gas_list, ranges, degree, od_path, coeff_path, low_res = 1e-2):
    for g_name in gas_list:
        logger.info(f'Gas: {g_name}')
        for i in range(len(ranges)-1):
            logger.info(f'Frequency window: {ranges[i]}-{ranges[i+1]}')
            ds = xr.open_dataset(f'{od_path}{g_name}/od_{g_name}_freq{ranges[i]}_{ranges[i+1]}_{low_res:.0e}.nc')
   #         T = ds.coords['DeltaT'].values
            ods = ds.od
#            errors = ds.error
 #           errors = xr.where(errors < 1e-8, 1e-8, errors)
            mask0 = ods.max(dim='DeltaT') > 1e-8
            ods = xr.where(mask0, ods, 0.)
            maskMAX = ods <= 690.7
            
            ods = ods.where(maskMAX, drop=False)
            #check if there are at least four valid points to fit, if not set to nan
            valid_points = maskMAX.sum(dim='DeltaT')
            ods = ods.where(valid_points >= degree + 1, drop=False)
            #where nan set to 690.7 to avoid issues in weighted fit
           
            name_out = f'{coeff_path}{g_name}/coeff_{degree}_{g_name}_freq{ranges[i]}_{ranges[i+1]}_{low_res:.0e}.nc'
            pp = ods.polyfit(dim='DeltaT', deg=degree)
            coeffs = pp.polyfit_coefficients
            mask = coeffs.isnull().any("degree")   # where polyfit failed
            custom = np.zeros((degree+1))  # your replacement coefficients
            custom[-1] = 690.7  # constant term set to 690.7
            custom_da = xr.DataArray(
                    custom,
                    dims=["degree"],
                    coords={"degree": coeffs["degree"]},
                ).broadcast_like(coeffs)
            coeffs = xr.where(mask, custom_da, coeffs)
                

            fitted = xr.polyval(ds.DeltaT, coeffs)
            ss_res = ((ods - fitted) ** 2).sum(dim='DeltaT')
            ss_tot = ((ods - ods.mean(dim='DeltaT')) ** 2).sum(dim='DeltaT')
            r_squared = 1 - ss_res / ss_tot

            dof = valid_points - (degree + 1)
            adjused_r_squared = 1 - (1 - r_squared) * (valid_points - 1) / dof
#            neg_mask = (fitted < 0).any(dim="DeltaT")
#            count_negative_points = neg_mask.sum()
#           tot = neg_mask.size
#           logger.info(f'NNeg_points = {count_negative_points.item()}/{tot}')
#            chi = (((ods - fitted)/errors) ** 2).sum(dim='DeltaT')
#            dof = 13 - degree - 1
#            chi = chi / dof

            ds = xr.Dataset({
                'coeff': coeffs,
                'mask0': mask0,
                'adj_r_squared': adjused_r_squared,
            })
            ds.to_netcdf(name_out)

    logger.info('Done')
        
        
    """
    if yerror:
                name_out = f'{coeff_path}{g_name}/coeff_{degree}_yer_{g_name}_freq{ranges[i]}_{ranges[i+1]}.nc'
                # Use xarray.apply_ufunc to vectorize over frequency and altitude
                coeffs = xr.apply_ufunc(
                    weighted_polyfit,
                    ods,
                    errors,
                    xr.DataArray(T, dims=['DeltaT']),  # broadcast T as an input
                    input_core_dims=[['DeltaT'], ['DeltaT'], ['DeltaT']],
                    output_core_dims=[['degree']],
                    vectorize=True,
                    kwargs={'degree': degree},
                    dask='parallelized',
                    output_dtypes=[float]
                )
                # Assign coordinate to degree dimension
                coeffs = coeffs.assign_coords(degree=np.arange(degree + 1)[::-1])  # Reverse to match np.polyfit order
            else:
"""