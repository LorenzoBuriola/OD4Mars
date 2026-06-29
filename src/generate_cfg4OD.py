import numpy as np
import PSGpy.cfg as cfg
import PSGpy.atm_obj as atm

def generate_OD_cfg(gas_list, ifile, opath):
    cfg_dict = cfg.read_cfg(ifile)
    for g_name in gas_list:
        temp = cfg_dict.copy()
        atmos = atm.atmosphere()
        atmos.get_atmosphere(temp)
        atmos.add_gas(g_name)
        atmos.continuum_list = ['Layering', 'Contributions']
        atmos.edit_cfg(temp)
        if g_name == 'H2O':
            temp['ATMOSPHERE-GAS'] = 'H2O,H2O,H2O'
            temp['ATMOSPHERE-UNIT'] = 'scl,scl,scl'
            temp['ATMOSPHERE-ABUN'] = '1,1,1'
            temp['ATMOSPHERE-TYPE'] = 'HIT[1:1],HIT[1:2],HIT[1:3]'
            temp['ATMOSPHERE-NGAS'] = 3
        cfg.dict_to_cfg(cfg_dict=temp,file_path=f'{opath}cfg_{g_name}.txt')