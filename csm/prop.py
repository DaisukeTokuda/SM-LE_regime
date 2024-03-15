from . import figure

DICT_UNIT = {
    'model': '-',
    'wilt': figure.SM_UNIT,
    'slope': f'{figure.LE_UNIT} / {figure.SM_UNIT}',
    'csm': figure.SM_UNIT,
    'lesat': figure.LE_UNIT
}
DICT_LONG_NAME = {
    'model': 'Best model index',
    'wilt': 'Wilting point',
    'slope': f'SM-LE slope',
    'csm': 'Critical soil moisture',
    'lesat': 'Saturated evaporation',
}


DICT_NCVAR_NAME = {
    'wilt': {
        '110': 'BPx_2Seg_LHSflat',
        '111': 'BPx1_3Seg',
    },
    'slope': {
        '010': 'Slope_1Seg',
        '110': 'RHSSlope_2Seg_LHSflat',
        '011': 'LHSSlope_2Seg_RHSflat',
        '111': 'MIDSlope_3Seg',
    },
    'csm': {
        '011': 'BPx_2Seg_RHSflat',
        '111': 'BPx2_3Seg',
    },
    'lesat': {
        '011': 'BPy_2Seg_RHSflat',
        '111': 'BPy2_3Seg',
    },
}

DICT_NETCDF_INFO = { # long_name
    'model': 'model index selected by BIC',
    'wilt': 'wilting point of SM (x value of the left break point)',
    'slope': 'linear slope of SM-LE',
    'csm': 'critical soil moisture (x value of the right break point)',
    'lesat': 'saturated LE (y value of the right break point)',
}