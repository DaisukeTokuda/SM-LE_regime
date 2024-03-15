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
from sklearn.model_selection import train_test_split
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

model_cond = {
    'experiment_id': [
        *ll.config.CPL_EXPERIMENT_IDS,
        *ll.config.OFL_EXPERIMENT_IDS,
    ],
    'source_id': ll.config.CPLOFL_SOURCE_IDS,
    'grid_label': '2x2',
    'time_range': '190301-201212',
}

model_type = 'lasso'
#ind_variable_ids = ['pr', 'rn']
ind_variable_ids = ['pr', 'rds']
#ind_variable_ids = ['pr', 'rsds']

prop_names = ll.PROP_NAMES

OVER_WRITE = True

# ===================================================================================================
def exe_each(model_sr):
    dict_model = ioutil.load_pickle(model_sr['path'], log=False)

    ret = []
    sim_key = [model_sr[k] for k in ll.config.SIM_ATTRS]
    for k, model in dict_model.items():
        coefs = model.coef_
        ret += [(*sim_key, *k, *coefs)]
    return ret

# ===================================================================================================
def main(*args):
    dir_name = ll.vs_atm.get_dir_name(model_type, ind_variable_ids)
    model_catalog = ll.catalog.read(dir_name)
    model_catalog = pdutil.filter(model_catalog, **model_cond)

    dst_path = ll.vs_atm.get_importance_path(dir_name)
    csv = pdutil.CSVUpdater(dst_path, ll.vs_atm.sens.COEF_COLUMNS + ind_variable_ids)

    for _, model_sr in model_catalog.iterrows():
        if not OVER_WRITE:
            _df = csv.find(**{k: model_sr[k] for k in ll.config.SIM_ATTRS})
            if len(_df) > 0: continue
        ret = exe_each(model_sr)
        for _ret in ret:
            csv.append(_ret)
    csv.save()
    return

# ===================================================================================================
if __name__=='__main__':
    main(*sys.argv)
