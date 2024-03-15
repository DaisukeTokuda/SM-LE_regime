#! /usr/bin/env python3
'''
Requirements:
    CSM property and CMIP climatology for the same period
'''
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import itertools
from concurrent import futures

from ncar import CMIP as cmip
import figlib
from util import stat as statutil
import dictutil
import pdutil
import ioutil

import lsmconst as ll

import warnings
warnings.simplefilter('ignore', RuntimeWarning)

bs_cond = {
    'experiment_id': [
        'historical',
        'land-hist', 'land-hist-cruNcep', 'land-hist-princeton', 'land-hist-wfdei',
    ],
    #'source_id': '~GISS-E2-1-G',
    'source_id': 'EC-Earth3-Veg',
    'grid_label': '2x2',
    'time_range': '190301-201212'
}

model_type = 'lasso'
#ind_variable_ids = ['pr', 'rn']
ind_variable_ids = ['pr', 'rds']
#ind_variable_ids = ['pr', 'rsds']

OVER_WRITE = False

# ===================================================================================================
def lasso_best(dst_path, smle_sr, ind_srs):
    '''
    Returns:
        ret ({(str, int, scalar): (model, int, tuple, scalar)}):
            {(prop_name, class_id): (alpha, model, min_bic)}
    '''
    prop_names = ll.PROP_NAMES
    data = ll.vs_atm.sens.encode_sr(smle_sr, ind_srs, prop_names=prop_names)
    # {(prop_name, class_id): [independent variables], dependent variable}
    ret = {}
    for k, (y, X) in data.items():
        ret[k] = ll.vs_atm.sens.select_best(y, X)
    ioutil.dump_pickle(dst_path, ret)
    return ret

# ===================================================================================================
def main(*args):
    smle_catalog = ll.catalog.read('prop_monthly')
    smle_catalog = pdutil.filter(smle_catalog, **bs_cond)
    smle_catalog = pdutil.filter(smle_catalog)
    #smle_catalog = pdutil.filter_by_group_size(smle_catalog, 'source_id', 10)

    cmip_catalog = cmip.catalog.read('local')
    cmip_catalog = cmip.catalog.filter(cmip_catalog,
        table_id='clim', **bs_cond)
    ind_catalogs = [
        pdutil.filter(cmip_catalog,
            variable_id=variable_id)
        for variable_id in ind_variable_ids
    ]

    sims = pdutil.get_overlapping(smle_catalog, *ind_catalogs,
        subset=ll.config.SIM_ATTRS)
    print(sims.shape)

    dst_dir_name = ll.vs_atm.get_dir_name(model_type, ind_variable_ids)

    args = []
    for _, sim in sims.iterrows():
        #if not os.path.exists(ll.path_overlap_mask(sim['source_id'])): continue
        #print(*{k: sim[k] for k in key_attrs}.values())
        dst_path = ll.file.get_file_path(dst_dir_name, sim, suffix='.pkl')
        if (not OVER_WRITE) and os.path.exists(dst_path): continue

        smle_sr = pdutil.filter(smle_catalog, log=False, **sim.to_dict()).iloc[0]
        ind_srs = [
            pdutil.filter(ind_catalog, log=False, **sim.to_dict()).iloc[0]
            for ind_catalog in ind_catalogs
        ]
        args += [(dst_path, smle_sr, ind_srs)]
        #break
    print(len(args))

    #with futures.ProcessPoolExecutor(max_workers=16) as executor:
    if True:
        for _args in args:
            #executor.submit(lasso_best, *_args)
            lasso_best(*_args)

# ===================================================================================================
if __name__=='__main__':
    #main(*sys.argv)
    dst_dir_name = ll.vs_atm.get_dir_name(model_type, ind_variable_ids)
    ll.catalog.write(dst_dir_name)
