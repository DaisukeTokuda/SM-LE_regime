import numpy as np
import xarray as xr
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from util import stat as statutil

from .. import config as llconf

DICT_VAR_MEAN = {
    'pr': 3.5e-05, # 3.532330387777989e-05
    'rn': 120, # 115.17374280895973
    'rds': 570, # 567.178782452478
    'rsds': 210 # 211.3761726431291
}
DICT_VAR_STD = {
    'pr': 2.7e-05, # 2.8410675303633106e-05
    'rn': 40, # 39.633440731038334
    'rds': 73,  # 72.1576242092533
    'rsds': 40, # 43.72784047346219
}

ALPHAS = []
for pow in range(-8, 8):
    for coef in [1, 2, 5]:
        ALPHAS += [(10 ** pow) * coef]

COEF_COLUMNS = llconf.SIM_ATTRS + ['prop_name', 'class_id']

# ===================================================================================================
def encode_sr(smle_sr, ind_srs, dict_prop_mask={}, prop_names=[]):
    '''
    Returns:
        data ({(str, str): ([np.ndarray, ...], np.ndarray)}):
            {(prop_name, class_id): (y, X)}
    '''
    smle_ds = xr.open_dataset(smle_sr['path'])
    dict_prop_smlearr = {k: smle_ds[k].to_masked_array() for k in prop_names}
    ind_arrs = []
    for ind_sr in ind_srs:
        arr = xr.open_dataset(ind_sr['path'])[ind_sr['variable_id']].to_masked_array()
        arr = np.ma.masked_greater(arr, 1e10)
        mean, std = DICT_VAR_MEAN[ind_sr['variable_id']], DICT_VAR_STD[ind_sr['variable_id']]
        arr = (arr - mean) / std
        ind_arrs += [arr]
    return encode(dict_prop_smlearr, ind_arrs, dict_prop_mask)


def encode(dict_k_arr, ind_maps, dict_k_class={}):
    '''
    Returns:
        data ({(str, str): ([np.ndarray, ...], np.ndarray)}):
            {(prop_name, class_id): (y, X)}
    '''
    default_class_map = np.zeros(ind_maps[0].shape, dtype='int')
    
    data = {}
    for k, y_map in dict_k_arr.items():
        has_y = ~(y_map.mask)
        class_map = dict_k_class.get(k, default_class_map)
        class_ids = sorted([val for val in np.unique(class_map) if val >= 0])

        for class_id in class_ids:
            class_mask = class_map == class_id
            mask = np.logical_and(has_y, class_mask)
            if mask.sum() == 0: continue

            y = y_map[mask]
            inds = [ind_map[mask] for ind_map in ind_maps]
            y, *inds = statutil.filter_invalid(y, *inds)
            #inds = [(ind - ind.mean()) / np.ma.std(ind) for ind in inds]
            X = np.ma.stack(inds, axis=0).T

            data[(k, class_id)] = (y, X)
    return data

# ===================================================================================================
def lasso_est(y, X, alphas):
    """
    Conduct Lasso regression for a single grid cell.

    Args:
        y (np.ndarray): The dependent variable samples for the grid cell.
        X (np.ndarray): The independent variable samples for the grid cell.
        alphas (List[float]): List of alphas for Lasso regression.
    Returns:
    Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        Tuple of coefficients for independent variables, standard errors,
        and a tuple containing W statistic values and their corresponding p-values for Shapiro-Wilk test.
    """
    if len(y) < 5:
        raise ValueError('less sample size:', len(y))

    models = []
    for alpha in alphas:
        model = Lasso(alpha=alpha)
        model.fit(X, y)
        models += [model]
    return models


def eval(model, y, y_pred):
    ll = statutil.log_likelihood(y, y_pred)
    num_params = 0
    coefs = model.coef_
    for coef in coefs:
        if coef != 0: num_params += 1
    bic = statutil.bic(ll, num_params, len(y))

    r2 = r2_score(y, y_pred)
    rmse = statutil.rmse(y, y_pred)
    return {'bic': bic, 'r2': r2, 'rmse': rmse}


def select_best(y, X):
    '''
    Returns:
        model
    '''
    train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
    #print(train_y.shape, test_y.shape)

    min_bic = 1e20
    best_model = None
    for alpha in ALPHAS:
        model = Lasso(alpha=alpha)
        model.fit(train_X, train_y)
        pred_y = model.predict(test_X)
        bic = eval(model, test_y, pred_y)['bic']
        if bic < min_bic:
            best_model = model
            min_bic = bic
    return best_model

