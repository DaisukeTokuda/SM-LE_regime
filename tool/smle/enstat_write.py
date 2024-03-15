#! /usr/bin/env python3
import os 
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

from ncar import CMIP as cmip

import lsmconst as ll

#src_dir_name = 'prop'
#dst_dir_name = 'prop_enstat'
src_dir_name = 'prop_monthly'
dst_dir_name = 'prop_enstat_monthly'

'''Target'''
dict_target = {
    'experiment_id': [
        *ll.config.OFL_EXPERIMENT_IDS,
    ],
    'source_id': [
        *ll.config.CPLOFL_SOURCE_IDS,
    ],
    'grid_label': '2x2',
    'time_range': '190301-201212',
}

ensemble_attr = 'experiment_id'
member_values = []

# ===================================================================================================
def calc_enstat(paths, var_names=None, log=True):
    '''
    Args:
        paths ([str,])
        var_names ([str,]): netCDF4 variable name
        log (bool)
    Returns:
        dict_enstat ({str: np.ma.MaskedArray})
        ds (xr.DataSet)
    '''
    dict_arr = None
    for path in paths:
        ds = xr.open_dataset(path)
        if var_names is None:
            var_names = [var_name for var_name in list(ds.variables) if not var_name in ('lat', 'lon')]
        if dict_arr is None:
            dict_arr = {var_name: [] for var_name in var_names}

        for var_name in var_names:
            dict_arr[var_name] += [ds[var_name].to_masked_array()]
        if log: print(path.replace(ll.config.BS_DIR, '').lstrip('/'))
    for var_name in var_names:
        dict_arr[var_name] = np.ma.stack(dict_arr[var_name], axis=0)
    #plt.imshow(dict_arr['csm'][0])
    #plt.show()
    #return

    dict_enstat = {}
    for var_name, arr in dict_arr.items():
        mean = arr.mean(axis=0)
        std = np.ma.std(arr, axis=0)
        dict_enstat[var_name + '_mean'] = mean
        dict_enstat[var_name + '_std'] = std
        dict_enstat[var_name + '_cv'] = std / mean
        num = (~(np.ma.getmaskarray(arr))).sum(axis=0).astype('int8')
        dict_enstat[var_name + '_num'] = np.ma.masked_less_equal(num, 0)
    return dict_enstat, ds

# ===================================================================================================
def main(*args):
    catalog = ll.catalog.read(src_dir_name)
    if not ensemble_attr in dict_target: dict_target.update({ensemble_attr: member_values})
    catalog = cmip.catalog.filter(catalog, log=True, **dict_target)

    ensembler = ll.catalog.Ensembler(ensemble_attr)
    for dict_attr, _catalog in ll.catalog.itersims(catalog, ensembler.other_sim_attr_names):
        dict_attr.update(ensembler.get_attr_values(_catalog))
        dst_path = ll.file.get_file_path(dst_dir_name, dict_attr)
        #print(dst_path)
        #continue

        dict_enstat, src_ds = calc_enstat(_catalog['path'], var_names=None, log=True)
        glob_attr = {
            'title': 'Ensemble properties',
            'samples': (',').join(_catalog[ensemble_attr].sort_values())
        }
        ll.nc.write(dst_path, dict_enstat, src_ds, glob_attr)
        print()
        #break
    return

if __name__=='__main__':
    main(*sys.argv)
    ll.catalog.write(dst_dir_name)