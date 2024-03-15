import os
import pandas as pd

import pdutil
from ncar import CMIP as cmip

from . import config as llconf
from . import file as llfile

# ===================================================================================================
def get_path(dir_name):
    return os.path.join(llconf.BS_DIR, 'out', 'catalog', dir_name + '.csv')


def read(dir_name, log=True):
    path = get_path(dir_name)
    df = pd.read_csv(path)
    if log: print(path, df.shape)
    return df

# ===================================================================================================
def filter_intersections_by_vars(catalog, *var_names, **kwargs):
    '''
    Args:
        catalog (pandas.DataFrame)
        var_names ([str])
        kwargs: for filter_intersections()
    Returns:
        (pandas.DataFrame)
    '''
    conds = []
    for var_name in var_names:
        table_id, variable_id = llconf.get_table_variable_ids(var_name)
        conds += [{'table_id': table_id, 'variable_id': variable_id}]
    return cmip.catalog.filter_intersections(catalog, conds, **kwargs)


def filter_member_num(catalog, ens_attr_name, min_members):
    '''
    Args:
        catalog (pandas.DataFrame)
        min_members (int)
    Returns:
        _catalog (pandas.DataFrame)
    '''
    has_members = [llfile.member2int(member_id) >= min_members
        for member_id in catalog[ens_attr_name]]
    _catalog = catalog[has_members]
    print('having enough members', _catalog.shape)
    return _catalog


def filter_single_member(catalog, log=True):
    _catalog = catalog.copy()
    attr_names = [attr for attr in llconf.SIM_ATTRS if not attr == 'member_id']
    #for attr_values, __catalog in catalog.groupby(attr_names):
    for attr_values, __catalog in catalog.groupby('source_id'):
        is_found = False
        #print(attr_values, __catalog.shape)
        for member_id in llconf.SINGLE_MEMBER_IDS:
            is_member = __catalog['member_id'] == member_id
            if is_member.sum() == 0: continue

            is_found = True
            _rm_catalog = __catalog[~is_member]
            break
        try:
            assert(is_found)
        except AssertionError as e:
            print(attr_values)
            print(__catalog)
            raise e
        _catalog.drop(_rm_catalog.index, inplace=True)
        #print(_rm_catalog.index, _catalog.shape)
    '''test'''
    #for _, _catalog in catalog.groupby(attr_names):
    #    assert(len(_catalog) == 1)
    if log: print(_catalog.shape)
    return _catalog

# ===================================================================================================
def get_overlapping_sims(*catalogs, subset=llconf.SIM_ATTRS):
    '''
    Args:
        catalog0 (pandas.DataFrame)
        catalog1 (pandas.DataFrame)
    Returns:
        (pandas.DataFrame)
    '''
    return pdutil.get_overlapping(*catalogs, subset=subset)


def itersims(catalog, attrs=None):
    '''
    Args:
        catalog (pandas.DataFrame)
        attrs ([str,])
    Yields:
        ({attr: value})
        _catalog (pandas.DataFrame)
    '''
    if attrs is None: attrs = catalog.columns.to_list()
    for values, _catalog in catalog.groupby(attrs):
        yield {k: v for k, v in zip(attrs, values)}, _catalog

# ===================================================================================================
# ensemble
# ===================================================================================================
class Ensembler:
    '''
    Attributes:
        attr_names ([str,])
    '''
    
    def __init__(self, attr_names):
        '''
        Args:
            attr_names (str or [str,])
        '''
        if isinstance(attr_names, str): attr_names = [attr_names]
        self.attr_names = attr_names

    @property
    def other_sim_attr_names(self):
        '''
        Returns:
            attr_names ([str,])
        '''
        return [attr_name for attr_name in llconf.SIM_ATTRS if not attr_name in self.attr_names]
    
    def get_path(self, dir_name, catalog, **kwargs):
        dict_attr = {}
        for attr_name in self.attr_names:
            dict_attr[attr_name] = str(len(catalog[attr_name].unique())) + 'members'
        kwargs.update(dict_attr)
        return llfile.get_file_path(dir_name, kwargs)

    def filter(self, catalog):
        return cmip.catalog.filter(catalog, **{attr_name: '*members' for attr_name in self.attr_names})


    def get_attr_values(self, catalog, years=None):
        '''
        Args:
            catalog (pd.DataFrame): Number of ensembles
            years (int)
        Returns:
            dict_attr ({str: str})
        '''
        dict_attr = {}
        for attr_name in self.attr_names:
            attr_value = str(len(catalog[attr_name].unique())) + 'members'
            if attr_name == 'time_range':
                attr_value += f'-{years}years'
            dict_attr[attr_name] = attr_value
        return dict_attr

# ===================================================================================================
def filter_time_range(catalog, years):
    '''
    Args:
        catalog (pandas.DataFrame)
        years (int): year length of time_range
    Returns:
        _catalog (pandas.DataFrame)
    '''
    has_members = []
    for time_range in catalog['time_range']:
        _years = int(time_range.split('-')[-1].rstrip('years'))
        has_members += [years == _years]
    _catalog = catalog[has_members]
    print(_catalog.shape)
    return _catalog


def get_member_num(attr_value):
    '''
    Args:
        attr_value (str)
    Returns:
        member_num (int)
    '''
    return int(attr_value.split('members')[0])

# ===================================================================================================
def get_multi_exp_sources(catalog, experiment_ids, log=True):
    '''
    Args:
        catalog (pandas.DataFrame)
        experiment_ids ([str])
        log (bool)
    Returns:
        source_ids (pandas.core.indexes.base.Index)
    '''
    _catalog = catalog[catalog['experiment_id'].isin(experiment_ids)]
    __catalog = _catalog.groupby('source_id')['experiment_id'].nunique()
    source_ids = __catalog.index[__catalog > 1]
    if log:
        print(len(source_ids), 'source_ids found')
        for source_id in source_ids:
            print(source_id, _catalog[_catalog['source_id'] == source_id]['experiment_id'].unique())
    return source_ids


def get_attrs_single_source_multi_experiments(catalog, source_id, experiment_ids):
    '''
    Args:
        calalog (pandas.DataFrame)
        source_id (str)
        experiment_ids ([str])
    Returns:
        attrs ([(str,)]): [(experiment_id, member_id, grid_label, path)]
    '''
    src_catalog = catalog[catalog['source_id'] == source_id]
    attrs = []
    for experiment_id in experiment_ids:
        src_exp_catalog = src_catalog[src_catalog['experiment_id'] == experiment_id]
        if src_exp_catalog.empty: continue

        for _, sr in src_exp_catalog.iterrows():
            member_id = sr['member_id']
            grid_label = sr['grid_label']
            path = sr['path']
            attrs += [(experiment_id, member_id, grid_label, path)]
    return attrs


def get_attrs_single_experiment(catalog, experiment_id):
    exp_catalog = catalog[
        (catalog['experiment_id'] == experiment_id)
    ]
    attrs = []
    for _, sr in exp_catalog.sort_values(by='path').iterrows():
        source_id = sr['source_id']
        member_id = sr['member_id']
        grid_label = sr['grid_label']
        path = sr['path']
        attrs += [(source_id, member_id, grid_label, path)]
    return attrs

# ===================================================================================================
def write(dir_name, is_pred=False):
    from pathlib import Path

    print('=== catalog ===')
    out_path = get_path(dir_name)

    def write_catalog(out_path, dir_name):
        root_path = os.path.join(llconf.BS_DIR, 'out', dir_name)
        root = Path(root_path)

        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, 'w') as f:
            if is_pred:
                attrs = ['model_' + k for k in llconf.SIM_ATTRS] + ['target_' + k for k in llconf.SIM_ATTRS]
            else:
                attrs = llconf.SIM_ATTRS
            text = ','.join(attrs)
            f.write(text + ',path\n')

            for _p in root.iterdir():
                if _p.is_dir(): continue
                texts = _p.stem.split('_')
                f.write((',').join(texts) + f',{str(_p)}\n')

    write_catalog(out_path, dir_name)
    df = pd.read_csv(out_path)
    df.sort_values(by='path').to_csv(out_path, index=False)
    print(out_path, df.shape)
