import os
import sys
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from util import stat as statutil

from .. import config as llconf

EVAL_COLUMNS = [
    'model_source_id', 'model_experiment_id', 
]

# ===================================================================================================
def get_pred_map_path(model_name, ind_variable_ids, source_id, experiment_id, member_id):
    dir_name = model_name + '_' + '-'.join(ind_variable_ids)
    file_name = '_'.join([source_id, experiment_id, member_id]) + '.nc'
    return os.path.join(llconf.BS_DIR, 'out', 'bpdef', dir_name, file_name)


def get_variable_name(prop_name, class_id, ind_variable_ids, scales):
    name_parts = []
    for variable_id, scale in zip(ind_variable_ids, scales):
        name_parts += [f'{variable_id}{scale:03d}']
    return '_'.join([prop_name, str(class_id), '-'.join(name_parts)])

def get_importance_path(model_name, ind_variable_ids, source_id, experiment_id, member_id):
    dir_name = model_name + '_' + '-'.join(ind_variable_ids)
    file_name = '_'.join([source_id, experiment_id, member_id]) + '_importance.pkl'
    return os.path.join(llconf.BS_DIR, 'out', 'tmp', 'bpdef', dir_name, file_name)

# ===================================================================================================
def encode_sr(smle_sr, ind_srs, ta_sr, keys):#, class_sr=None):
    smle_ds = xr.open_dataset(smle_sr['path'])
    dict_k_map = {
        key: smle_ds[key].to_masked_array() for key in keys
    }
    base_mask = ~(smle_ds['model'].to_masked_array().mask) # ocean, too low

    ind_maps = [
        xr.open_dataset(ind_sr['path'])[ind_sr['variable_id']].to_masked_array()
        for ind_sr in ind_srs
    ]

    ta_map = xr.open_dataset(ta_sr['path'])[ta_sr['variable_id']].to_masked_array()
    base_mask = np.logical_and(base_mask, ta_map >= llconf.TAIR_MIN)
    dict_class_mask = {0: base_mask}

    return encode(dict_k_map, ind_maps, dict_class_mask)


def encode(dict_k_map, ind_maps, dict_class_mask):
    '''
    Returns:
        ret {(scalar, any): (np.ndarray, np.ndarray)}:
            {(prop_name, class_id): (X, y)}
    '''
    ret = {}
    for k, dep_map in dict_k_map.items():
        has_prop = ~(dep_map.mask)

        for class_key, class_mask in dict_class_mask.items():
            if class_mask.sum() == 0: continue
            is_class_and_has_prop_mask = np.logical_and(has_prop, class_mask)
            is_class_and_not_prop_mask = np.logical_and(~has_prop, class_mask)
            if is_class_and_has_prop_mask.sum() == 0: continue
            if is_class_and_not_prop_mask.sum() == 0: continue

            ind_arrs = []
            for ind_map in ind_maps:
                _ind_arr_has_prop = ind_map[is_class_and_has_prop_mask]
                _ind_arr_not_prop = ind_map[is_class_and_not_prop_mask]
                ind_arr = np.ma.concatenate((_ind_arr_has_prop, _ind_arr_not_prop), axis=0)
                ind_arrs += [ind_arr]

            y = np.array([1] * is_class_and_has_prop_mask.sum() + [0] * is_class_and_not_prop_mask.sum())
            y, *ind_arrs = statutil.filter_invalid(y, *ind_arrs)
            if len(y) == 0: continue
            X = np.ma.stack(ind_arrs, axis=1)
            print(X.shape, y.shape, is_class_and_has_prop_mask.sum(), is_class_and_not_prop_mask.sum())

            ret[(k, class_key)] = (X, y)
    return ret
