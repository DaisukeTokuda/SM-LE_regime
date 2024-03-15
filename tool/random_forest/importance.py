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
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance

import dictutil
import pdutil
import ioutil
from ncar import CMIP as cmip

import lsmconst as ll

#experiment_ids = ['land-hist', 'land-hist-cruNcep', 'land-hist-princeton', 'land-hist-wfdei']
#source_ids = ['IPSL-CM6A-LR', 'MIROC6']
model_cond = {
    'experiment_id': [
        *ll.config.CPL_EXPERIMENT_IDS,
        *ll.config.OFL_EXPERIMENT_IDS,
    ],
    'source_id': ll.config.CPLOFL_SOURCE_IDS,
    'grid_label': '2x2',
    'time_range': '190301-201212',
}
target_cond = {
    'experiment_id': [
        *ll.config.CPL_EXPERIMENT_IDS,
        *ll.config.OFL_EXPERIMENT_IDS,
    ],
    'source_id': ll.config.CPLOFL_SOURCE_IDS,
    'grid_label': '2x2',
    'time_range': '190301-201212',
}

prop_names = ll.PROP_NAMES[:2]
#ind_variable_ids = ['pr', 'rn']
ind_variable_ids = ['pr', 'rds']
#ind_variable_ids = ['pr', 'rsds']

model_type = ll.stat.bpdef.MODEL_TYPES[2]

match_attr_names = ['source_id', 'experiment_id', 'member_id']#[:1]
OVER_WRITE = False

scorers = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
dict_score_stat = {
    'mean': 'importances_mean',
    'std': 'importances_std',
}

# ===================================================================================================
def exe_each_sim(model_sim, model_sr, smle_sr, ind_srs, ta_sr):
    dict_model = ioutil.load_pickle(model_sr['path'], log=False)
    dict_prop_Xy = ll.stat.bpdef.encode_sr(
        smle_sr, ind_srs, ta_sr, prop_names
    )

    ret = []
    sim_key = list(model_sim.to_dict().values())
    for key, model in dict_model.items():
        # key: (prop_name, class_id)
        X, y = dict_prop_Xy[key]
        _, test_X, _, test_y = train_test_split(X, y, random_state=0)
        dict_result = permutation_importance(model, test_X, test_y, n_repeats=30, random_state=0, scoring=scorers)
        for scorer in scorers:
            for stat_name, stat_key in dict_score_stat.items():
                for variable_id, value in zip(ind_variable_ids, dict_result[scorer][stat_key]):
                    ret += [(*sim_key, *key, variable_id, scorer, stat_name, value)]
    return ret


def has_all_pairs(df):
    for ind_variable_id, stat_name, type in itertools.product(ind_variable_ids, scorers, ['mean', 'std']):
        _df = pdutil.filter(df, log=False,
            variable_id=ind_variable_id, stat_name=stat_name, type=type)
        if len(_df) == 0: return False
    return True

# ===================================================================================================
def main(*args):
    dir_name = ll.vs_atm.get_dir_name(model_type, ind_variable_ids)
    model_catalog = ll.catalog.read(dir_name)
    model_catalog = pdutil.filter(model_catalog, **model_cond)
    model_sims = model_catalog[ll.config.SIM_ATTRS]

    smle_catalog = ll.catalog.read('prop_monthly')
    smle_catalog = cmip.catalog.filter(smle_catalog, **target_cond)
    target_sims = smle_catalog[ll.config.SIM_ATTRS]

    cmip_catalog = cmip.catalog.read('local')
    subst_catalog = pdutil.filter(cmip_catalog,
        institution_id='forcing', table_id='clim')
    cmip_catalog = pdutil.filter(cmip_catalog,
        table_id='clim', **target_cond)

    dst_path = ll.vs_atm.get_importance_path(dir_name)
    csv = pdutil.CSVUpdater(dst_path, ll.vs_atm.IMPORTANCE_COLUMNS)

    args = []
    for _, model_sim in model_sims.iterrows():
        #print(*sim.values)
        _model_catalog = pdutil.filter(model_catalog, log=False, **model_sim)
        assert(len(_model_catalog) == 1)
        _target_cond = {k: model_sim[k] for k in match_attr_names}
        _target_sims = pdutil.filter(target_sims, log=False, **_target_cond)

        for _, target_sim in _target_sims.iterrows():
            if not OVER_WRITE:
                cond = model_sim.to_dict()
                _df = csv.find(**cond)
                if has_all_pairs(_df): continue

            #    if len(_df) > 0: continue
            smle_sr = pdutil.filter(smle_catalog, log=False, **target_sim)
            assert(len(smle_sr) == 1)
            smle_sr = smle_sr.iloc[0]

            try:
                ind_srs = ll.vs_atm.find_sr(cmip_catalog, subst_catalog, target_sim, ind_variable_ids)
            except ValueError:
                continue

            try:
                tas_sr = ll.vs_atm.find_sr(cmip_catalog, subst_catalog, target_sim, 'tas')
            except ValueError:
                continue

            args += [(
                model_sim, _model_catalog.iloc[0], smle_sr, ind_srs, tas_sr
            )]
        #break
    print(len(args))
    #exe_each_sim(*args[0])
    #return

    fs = []
    with futures.ProcessPoolExecutor(max_workers=16) as executor:
        for _args in args:
            #fs += [_main_each_sim(*_args)]
            fs += [executor.submit(exe_each_sim, *_args)]
            #break

    for f in fs:
        for res in f.result():
            csv.append(res)
    csv.save()

# ===================================================================================================
if __name__=='__main__':
    main(*sys.argv)
