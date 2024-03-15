FILL_VALUE = 1e20
__DICT_DTYPE_FILVAL = {
    'int8': -9,
    'i': -9999,
    'f': 1e20,
}
DICT_NETCDF_INFO = { # (standard_name, long_name)
    ('001', 'param', 0): ('Const_1Seg', 'Constant of the 1 segment'),
    ('010', 'param', 0): ('Intercept_1Seg', 'Interception of the 1 segment linear regression'),
    ('010', 'param', 1): ('Slope_1Seg', 'Slope of the 1 segment linear regression'),
    ('110', 'param', 0): ('BPx_2Seg_LHSflat', 'SM Break Point of the 2 segment LHS-flat linear regression'),
    ('110', 'param', 1): ('BPy_2Seg_LHSflat', 'y Break Point of the 2 segment LHS-flat linear regression'),
    ('110', 'param', 2): ('RHSSlope_2Seg_LHSflat', 'RHS Slope of the 2 segment LHS-flat segment linear regression'),
    ('011', 'param', 0): ('BPx_2Seg_RHSflat', 'SM Break Point of the 2 segment RHS-flat linear regression'),
    ('011', 'param', 1): ('BPy_2Seg_RHSflat', 'y Break Point of the 2 segment RHS-flat linear regression'),
    ('011', 'param', 2): ('LHSSlope_2Seg_RHSflat', 'LHS Slope of the 2 segment RHS-flat linear regression'),
    ('111', 'param', 0): ('BPx1_3Seg', 'First SM Break Point of the 3 segment linear regression'),
    ('111', 'param', 1): ('BPy1_3Seg', 'First LE Break Point of the 3 segment linear regression'),
    ('111', 'param', 2): ('MIDSlope_3Seg', 'Mid Slope of the 3 segment linear regression'),
    ('111', 'param', 3): ('BPx2_3Seg', 'Second SM Break Point of the 3 segment linear regression'),
    ('111', 'param', 4): ('BPy2_3Seg', 'Second LE Break Point of the 3 segment linear regression'),
    ('001', 'error', 0): ('RSS_001', 'Residual sum of square (001)'),
    ('010', 'error', 0): ('RSS_010', 'Residual sum of square (010)'),
    ('110', 'error', 0): ('RSS_110', 'Residual sum of square (110)'),
    ('011', 'error', 0): ('RSS_011', 'Residual sum of square (011)'),
    ('111', 'error', 0): ('RSS_111', 'Residual sum of square (111)'),
    ('001', 'error', 1): ('BIC_001', 'Bayesian Information Criteria in terms of RSS (001)'),
    ('010', 'error', 1): ('BIC_010', 'Bayesian Information Criteria in terms of RSS (010)'),
    ('110', 'error', 1): ('BIC_110', 'Bayesian Information Criteria in terms of RSS (110)'),
    ('011', 'error', 1): ('BIC_011', 'Bayesian Information Criteria in terms of RSS (011)'),
    ('111', 'error', 1): ('BIC_111', 'Bayesian Information Criteria in terms of RSS (111)'),
    'DOF': ('DOF_SM', 'Degree of freedom of soil moisture'),
}

def get_fill_value(dtype):
    '''
    Args:
        dtype (numpy.dtype or str)
    Returns:
        (scalar)
    '''
    for k, v in __DICT_DTYPE_FILVAL.items():
        if str(dtype).startswith(k): return v