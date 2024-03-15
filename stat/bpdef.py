import numpy as np
import xarray as xr
import scipy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, ShuffleSplit
import warnings
warnings.simplefilter('ignore')
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
from util import stat as statutil

from .. import config

__all__ = [
    'MODEL_NAMES',
    'encode',
    'encode_sr',
    'est',
    'est_best',
    'eval',
]

MODEL_TYPES = ['logistic', 'decisiontree', 'randomforest']

DICT_SCORE_LONGNAME = {
    'f1': 'F1 score',
    'roc_auc': 'AUC-ROC'
}

# ===================================================================================================
def est(X, y, model_type='logistic', **kwargs):
    '''
    Args:
        X (np.ndarray [n_samples, n_features])
        y (np.ndarray [n_samples,])
    Returns:
        coef ((scalar,)) (n_features,)
        intc (scalar)
        y_pred (np.ndarray)
        y_score (np.ndarray)
    '''
    model = {
        'logistic': LogisticRegression(**kwargs),
        'decisiontree': DecisionTreeClassifier(criterion='gini', max_depth=4, random_state=1),
        'randomforest': RandomForestClassifier(criterion='gini', n_estimators=25, random_state=1, n_jobs=1),
    }[model_type]
    model.fit(X, y)
    #coef = model.coef_[0] # (n_features,)
    #intc = model.intercept_[0] # scalar
    return model


def est_best(X, y, model_type='logistic', **kwargs):
    '''
    Args:
        X (np.ndarray [n_samples, n_features])
        y (np.ndarray [n_samples,])
    Returns:
        coef ((scalar,)) (n_features,)
        intc (scalar)
        y_pred (np.ndarray)
        y_score (np.ndarray)
    '''
    #train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=0)
    shuffle_split = ShuffleSplit(n_splits=1, test_size=.25, random_state=0)
    dict_estimator = {
        'logistic': LogisticRegression,
        'decisiontree': DecisionTreeClassifier,
        'randomforest': RandomForestClassifier,
    }
    dict_param_grid = {
        'logistic': {
            'penalty': ['l1', 'l2', 'elasticnet', None],
            'C': [10 ** i for i in range(-5, 6)],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
            'max_iter': [50, 100, 200, 300],
            'random_state': [0],
            #'random_state': [i for i in range(0, 101)],
        },
        'decisiontree': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [i for i in range(1, 11)],
            'min_samples_split': [2, 5, 10, 20, 30, 40, 50],
            'min_samples_leaf': [1, 2, 4, 8, 16, 32],
            'max_features': ['sqrt', 'log2', None],
            'min_impurity_decrease': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
            #'random_state': [i for i in range(0, 101)],
            'random_state': [0],
        },
        'randomforest': {
            #'n_estimators': [i for i in range(1, 21)] + [50, 100, 200],
            'n_estimators': [1, 2, 5, 10, 20, 50],
            'criterion': ['gini', 'entropy'],
            'max_depth':[None, 1, 2, 3, 4, 5, 10, 20],# 30, 40, 50],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'random_state': [0],
            #'random_state': [i for i in range(0, 101)],
        },
    }
    dict_param_random = {
        'logistic': {
            'C': scipy.stats.uniform(0.00001, 1000),
            'random_state': scipy.stats.randint(0, 100),
        },
    }

    estimator = dict_estimator[model_type]()
    param = dict_param_grid[model_type]
    n_jobs = 1 if model_type == 'logistic' else 16
    clf = GridSearchCV(estimator, param,
        n_jobs=n_jobs, cv=shuffle_split, scoring='roc_auc')
    clf.fit(X, y)
    #clf.fit(train_X, train_y)
    #pred_y = clf.predict(test_X)
    #score = f1_score(test_y, pred_y, average='micro')
    return clf.best_estimator_


def pred(model, X):
    y_pred = model.predict(X)
    y_score = model.predict_proba(X)[:, 1]
    return y_pred, y_score


def eval(y, y_pred, y_score):
    res = {}
    res['y_1'] = (y == 1).sum()
    res['y_0'] = (y == 0).sum()
    res['ypred_1'] = (y_pred == 1).sum()
    res['ypred_0'] = (y_pred == 0).sum()
    res['accuracy'] = accuracy_score(y_true=y, y_pred=y_pred)
    res['precision'] = precision_score(y_true=y, y_pred=y_pred)
    res['recall'] = recall_score(y_true=y, y_pred=y_pred)
    res['f1'] = f1_score(y_true=y, y_pred=y_pred)
    res['roc_auc'] = roc_auc_score(y_true=y, y_score=y_score)
    return res
