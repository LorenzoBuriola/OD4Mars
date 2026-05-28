import logging
import numpy as np
import xarray as xr
import subprocess

logger = logging.getLogger(__name__)

def input4pack(gas_list, ranges, degree, coeff_path, out_path, low_res):

    if low_res == 1e-3:
        v1 = 90.0005
        v2 = 480.0005
    elif low_res == 1e-4:
        v1 = 330.
        v2 = 360.
    else:
        low_res = 1e-2
        v1 = 100.
        v2 = 3000.

    nn = int((v2-v1)/low_res) +1
    vv = np.arange(v1,v2+low_res,low_res)
    #nn = int((ranges[-1]-ranges[0])/low_res)
    layers = 55

    eps = 1e-5

    for ii,gas in enumerate(gas_list):
        logger.info(f'Processing {gas}')
        coeff_list =[]
        for rr in ranges[:-1]:
            coeff = xr.open_dataset(f'{coeff_path}{gas}/coeff_{degree}_{gas}_freq{rr}_{int(rr+40)}_{low_res:.0e}.nc')
            coeff_list.append(coeff)
        coeff_all = xr.concat(coeff_list, dim='freq')
        coeff_all = coeff_all.sortby('freq').sel(freq=slice(v1 - eps, v2 + eps))
        #check dimension
        if coeff_all.sizes['freq'] != nn:
            logger.warning(f'ERROR, size mismatch, nn = {coeff_all.sizes['freq']}, expected = {nn}')
        coeff_all = coeff_all.transpose('freq', 'altitude', 'degree')
        nquad = coeff_all.mask0.sum(dim = 'freq').values

        ind = []
        for jj in range(layers):
            ind.append(np.argwhere(coeff_all.mask0.values[:, jj]).flatten())

        cind = coeff_all.coeff.values

        for jj in range(layers):
            namefile = f'c{ii+1:02d}{jj+1:03d}'
            with open(out_path / namefile, 'wb') as fid1:
                fid1.write(np.array(nn, dtype=np.int32).tobytes())                              #npn(layer)
                fid1.write(np.array(nquad[jj], dtype=np.int32).tobytes())                       #nquad
                fid1.write(np.array(ind[jj]+1, dtype=np.int32).tobytes()) # Fortran indexing    #ind
                fid1.write(np.array(vv[ind[jj]], dtype=np.float64).tobytes())                   #vind
                for kk in reversed(range(degree+1)):
                    fid1.write(np.array(cind[ind[jj],jj,kk], dtype=np.float64).tobytes())       #cc

def run_packoneband(exe_path, name_database, degree, outfile):
    cmd = ['./pack_oneband.out', name_database, str(degree)]
    try:
        with open(outfile, 'w') as f:
            result = subprocess.run(cmd, 
                                    cwd=exe_path,
                                    stdout=f,
                                    stderr=subprocess.PIPE,
                                    text=True
                                )
        if result.returncode != 0:
            logger.error(f'Error running packoneband: {result.stderr}')
        else:
            logger.info(f'packoneband finished successfully. Output in {outfile}')
    except Exception:
        logger.exception("Failed to execute ./packoneband.out")