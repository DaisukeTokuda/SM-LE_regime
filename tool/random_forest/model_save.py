#! /usr/bin/env python3
'''
Requirements:
    CSM property and CMIP climatology for the same period
'''
import os
import sys
import numpy as np
from concurrent import futures

import ioutil
import pdutil
from ncar import CMIP as cmip
from util.profile import stopwatch

import lsmconst as ll

bs_cond = {
    #'source_id': '~GISS-E2-1-G',
    'source_id': [
        'ACCESS-ESM1-5',
        'CESM2',
        'CMCC-CM2-SR5',
        'CNRM-CM6-1',
        'CNRM-ESM2-1',
        'CanESM5',
        'EC-Earth3',
        'EC-Earth3-Veg',
        'GISS-E2-1-G',
        'GISS-E2-1-H',
        'HadGEM3-GC31-LL',
        'INM-CM5-0',
        'IPSL-CM6A-LR',
        'MIROC-ES2L',
        'MIROC6',
        'MPI-ESM1-2-HR',
        'MPI-ESM1-2-LR',
        'MRI-ESM2-0',
        'UKESM1-0-LL',
    ],
    'experiment_id': [
        'historical',
        'land-hist', 'land-hist-cruNcep', 'land-hist-princeton', 'land-hist-wfdei',
    ],
    'grid_label': '2x2',
    'time_range': '190301-201212',
}

prop_names = ll.PROP_NAMES[:2]
#ind_variable_ids = ['pr', 'rn']
ind_variable_ids = ['pr', 'rds']
#ind_variable_ids = ['pr', 'rsds']

model_type = ll.stat.bpdef.MODEL_TYPES[2]
dst_dir_name = ll.vs_atm.get_dir_name(model_type, ind_variable_ids)

# ===================================================================================================
@stopwatch
def exe_process(dst_path, smle_sr, ind_srs, ta_sr):
    dict_prop_Xy = ll.stat.bpdef.encode_sr(
        smle_sr, ind_srs, ta_sr, prop_names
    )
    print(dict_prop_Xy.keys())

    dict_model = {}
    for k, (X, y) in dict_prop_Xy.items():
        #model = ll.stat.bpdef.est(X, y, model_type)
        dict_model[k] = ll.stat.bpdef.est_best(X, y, model_type)
    ioutil.dump_pickle(dst_path, dict_model)
    return
# ===================================================================================================
def main(*args):
    smle_catalog = ll.catalog.read('prop_monthly')
    smle_catalog = pdutil.filter(smle_catalog,
        **bs_cond)

    cmip_catalog = cmip.catalog.read('local')
    ind_catalogs = [
        pdutil.filter(cmip_catalog,
            table_id='clim', variable_id=ind_variable_id, **bs_cond)
        for ind_variable_id in ind_variable_ids
    ]

    ta_catalog = cmip.catalog.filter(cmip_catalog,
        table_id='clim', variable_id='tas', **bs_cond)

    sims = ll.catalog.get_overlapping_sims(
        smle_catalog, *ind_catalogs, ta_catalog,
    )
    print(sims.shape)

    dst_dir_name = ll.vs_atm.get_dir_name(model_type, ind_variable_ids)

    args = []
    for _, sim in sims.iterrows():
        smle_sr = pdutil.filter(smle_catalog, log=False, **sim.to_dict()).iloc[0]
        ind_srs = [
            pdutil.filter(ind_catalog, log=False, **sim.to_dict()).iloc[0]
            for ind_catalog in ind_catalogs
        ]
        ta_sr = pdutil.filter(ta_catalog, log=False, **sim.to_dict()).iloc[0]
        dst_path = ll.file.get_file_path(dst_dir_name, sim, suffix='.pkl')

        args += [(dst_path, smle_sr, ind_srs, ta_sr)]
        #break
    print(len(args))

    max_workers = 16 if model_type == 'logistic' else 2
    #with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    if True:
        for _args in args:
            #executor.submit(exe_process, *_args)
            exe_process(*_args)
            #fs += [exe_process(bs_key, smle_sr, ind_srs, ta_sr)]
            #break

# ===================================================================================================
if __name__=='__main__':
    #main(*sys.argv)
    ll.catalog.write(dst_dir_name)