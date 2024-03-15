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

import dictutil
import pdutil
import ioutil
from ncar import CMIP as cmip

import lsmconst as ll

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

file_name = model_type + '_' + '-'.join(ind_variable_ids)

match_attr_names = ['source_id', 'experiment_id', 'member_id']#[:1]
OVER_WRITE = False

scorers = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# ===================================================================================================
def exe_each_sim(model_sim, target_sim, model_sr, smle_sr, ind_srs, ta_sr):
    dict_model = ioutil.load_pickle(model_sr['path'], log=False)
    dict_prop_Xy = ll.stat.bpdef.encode_sr(
        smle_sr, ind_srs, ta_sr, prop_names
    )

    ret = {}
    for key, model in dict_model.items():
        # key: (source_id, experiment_id, member_id, (prop_name, class_id))
        X, y = dict_prop_Xy[key]
        _, test_X, _, test_y = train_test_split(X, y, random_state=0)
        y_pred, y_score = ll.stat.bpdef.pred(model, test_X)
        _ret = ll.stat.bpdef.eval(test_y, y_pred, y_score)
        ret.update(dictutil.add_key(_ret, *[model_sim[k] for k in ll.config.SIM_ATTRS], *key, *[target_sim[k] for k in ll.config.SIM_ATTRS]))
    return ret

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

    dst_path = ll.vs_atm.get_score_path(dir_name)
    csv = pdutil.CSVUpdater(dst_path, ll.vs_atm.SCORE_COLUMNS)

    args = []
    for _, model_sim in model_sims.iterrows():
        #print(*sim.values)
        _model_catalog = pdutil.filter(model_catalog, log=False, **model_sim)
        assert(len(_model_catalog) == 1)
        _target_cond = {k: model_sim[k] for k in match_attr_names}
        _target_sims = pdutil.filter(target_sims, log=False, **_target_cond)

        for _, target_sim in _target_sims.iterrows():
            if not OVER_WRITE:
                cond = {'model_' + k: v for k, v in model_sim.to_dict().items()}
                cond.update({'target_' + k: v for k, v in target_sim.to_dict().items()})
                _df = csv.find(**cond)
                if len(_df) > 0: continue
            smle_sr = pdutil.filter(smle_catalog, log=False, **target_sim)
            assert(len(smle_sr) == 1)
            smle_sr = smle_sr.iloc[0]

            try:
                ind_srs = ll.vs_atm.find_sr(cmip_catalog, subst_catalog, ind_variable_ids)
            except ValueError:
                continue

            try:
                tas_sr = ll.vs_atm.find_sr(cmip_catalog, subst_catalog, 'tas')
            except ValueError:
                continue

            args += [(
                model_sim, target_sim, _model_catalog.iloc[0], smle_sr, ind_srs, tas_sr
            )]

        #break
    print(len(args))
    #dict_ret = exe_each_sim(*args[0])
    #for k, v in dict_ret.items():
    #    print([*k, v])
    #return

    fs = []
    with futures.ProcessPoolExecutor(max_workers=16) as executor:
        for _args in args:
            #fs += [_main_each_sim(*_args)]
            fs += [executor.submit(exe_each_sim, *_args)]
            #break

    data = []
    for f in fs:
        for k, v in f.result().items():
            data += [(*k, v)]
    for _data in data:
        csv.append(_data)
    csv.save()

# ===================================================================================================
if __name__=='__main__':
    main(*sys.argv)
