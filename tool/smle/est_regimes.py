#! /usr/bin/env python3
import os
import sys
import numpy as np
import xarray as xr
import datetime as dt
import pandas as pd

from ncar import CMIP as cmip

import lsmconst as ll

bs_cond = {
    'experiment_id': [
        'land-hist', 'land-hist-cruNcep', 'land-hist-wfdei', 'land-hist-princeton',
        'historical',
    ],
    'grid_label': '2x2',
}

time_ranges = [ # (start_year, end_year)
    (1903, 2012),
]

sm_scale = 1e-2
le_scale = None

dir_name = 'csm_monthly'
#over_write = False
over_write = True

# ===================================================================================================
def main(*args):
    catalog = cmip.catalog.read('local')
    catalog = cmip.catalog.filter(catalog, **bs_cond)

    (sm_table, sm_variable), (le_table, le_variable) = ll.config.get_table_variable_ids('sm', 'le')
    conds = [
        {'table_id': sm_table, 'variable_id': sm_variable},
        {'table_id': le_table, 'variable_id': le_variable},
    ]
    intersections = cmip.catalog.filter_intersections(catalog, conds)

    for _, sim in intersections.iterrows():
        _catalog = cmip.catalog.filter(catalog, log=False, **sim.to_dict())

        for start_year, end_year in time_ranges:
            time_range = cmip.dtutil.ints2tr(start_year, 1, end_year, 12)
            out_path = ll.file.get_file_path(dir_name, sim, time_range=str(time_range))
            if (not over_write) and os.path.exists(out_path): continue

            try:
                sm_ds = cmip.read_nc(_catalog, time_range,
                    table_id=sm_table, variable_id=sm_variable)
            except:
                continue
            try:
                le_ds = cmip.read_nc(_catalog, time_range, 
                    table_id=le_table, variable_id=le_variable)
            except:
                continue
            sm_arr = sm_ds[sm_variable].to_masked_array()
            if sm_scale is not None: sm_arr *= sm_scale
            le_arr = le_ds[le_variable].to_masked_array()
            if le_scale is not None: le_arr *= le_scale
            print(sm_arr.shape, le_arr.shape)
            assert(len(sm_arr) == len(le_arr))
            del le_ds

            dict_res = ll.est_models(sm_arr, le_arr)
            ll.nc.write_csm(out_path, dict_res, sm_ds, dim_names=sm_ds[sm_variable].dims[1:])
            del sm_ds
            #return
    return

if __name__=='__main__':
    main(*sys.argv)
    #ll.catalog.write(dir_name)