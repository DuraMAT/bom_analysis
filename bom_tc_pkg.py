import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
pd.set_option('display.max_columns', 500)

import seaborn as sns
sns.set_theme(font_scale=1)

import os
from pathlib import Path
from tqdm import tqdm
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

import pickle
import joblib

import shap
import pingouin as pg

from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.svm import SVR

from scipy.interpolate import UnivariateSpline

def store_metrics(y_train, y_train_pred, y_test, y_test_pred, record, model_name):
    record[model_name]['y_train_pred'] = y_train_pred

    record[model_name]['y_test_pred'] = y_test_pred

    rmse_train = mean_squared_error(y_train, y_train_pred, squared=False)
    rmse_test = mean_squared_error(y_test, y_test_pred, squared=False)

    mape_train = mean_absolute_percentage_error(y_train, y_train_pred)
    mape_test = mean_absolute_percentage_error(y_test, y_test_pred)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    mae_train = mean_absolute_error(y_train, y_train_pred)
    mae_test = mean_absolute_error(y_test, y_test_pred)

    record[model_name]['rmse_train'] = rmse_train
    record[model_name]['rmse_test'] = rmse_test

    record[model_name]['mape_train'] = mape_train
    record[model_name]['mape_test'] = mape_test

    record[model_name]['r2_train'] = r2_train
    record[model_name]['r2_test'] = r2_test

    record[model_name]['mae_train'] = mae_train
    record[model_name]['mae_test'] = mae_test

def extract_metrics(records):
    result = {
        'model': [],
        'train_test': [],
        'rmse': [],
        'r2': [],
        'mae': [],
        'mape': [],
        'cv_rmse_mean': [],
        'cv_rmse_std': []
    }
    for model_name in records.keys():
        # print(model_name)
        result['model'].append(model_name)
        result['train_test'].append('train')
        result['rmse'].append(records[model_name]['rmse_train'])
        result['r2'].append(records[model_name]['r2_train'])
        result['mae'].append(records[model_name]['mae_train'])
        result['mape'].append(records[model_name]['mape_train'])
        result['cv_rmse_mean'].append(records[model_name]['cv_mean'])
        result['cv_rmse_std'].append(records[model_name]['cv_std'])

        result['model'].append(model_name)
        result['train_test'].append('test')
        result['rmse'].append(records[model_name]['rmse_test'])
        result['r2'].append(records[model_name]['r2_test'])
        result['mae'].append(records[model_name]['mae_test'])
        result['mape'].append(records[model_name]['mape_test'])
        result['cv_rmse_mean'].append(np.nan)
        result['cv_rmse_std'].append(np.nan)

    return pd.DataFrame(result)

def gridsearch(model, param_grid, X_train, y_train, record, model_name, scoring='neg_root_mean_squared_error'):
    grid_search = GridSearchCV(estimator = model, param_grid = param_grid, n_jobs=-1,
                        cv = 5, scoring=scoring)

    grid_search.fit(X_train, y_train)

    record[model_name]['best_params'] = grid_search.best_params_
    
    df = pd.DataFrame(grid_search.cv_results_)
    best_inx = grid_search.best_index_

    # sklearn gridsearch use negative rmse as score
    record[model_name]['cv_scores'] = -1 * df.loc[best_inx, 'split0_test_score': 'split4_test_score'].to_numpy()
    record[model_name]['cv_mean'] = df.loc[best_inx, 'mean_test_score'] * -1
    record[model_name]['cv_std'] = df.loc[best_inx, 'std_test_score']

    return grid_search


def train_ols(X_train, y_train, X_test, y_test, record, model_name='ols'):
    model = LinearRegression()

    record[model_name]['best_params'] = None
    # cv of 5

    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    cv_mean = cv_scores.mean() * -1
    cv_std = cv_scores.std()

    record[model_name]['cv_scores'] = cv_scores * -1
    record[model_name]['cv_mean'] = cv_mean
    record[model_name]['cv_std'] = cv_std

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    store_metrics(y_train, y_train_pred, y_test, y_test_pred, record, model_name)

    return model


def train_ridge(X_train, y_train, X_test, y_test, record, model_name='ridge'):
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100, 1000],
        'solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
    }

    ridge = Ridge(random_state=12)

    grid_search = gridsearch(ridge, param_grid, X_train, y_train, record, model_name)

    ridge = Ridge(random_state=12, **grid_search.best_params_)
    ridge.fit(X_train, y_train)
    y_train_pred = ridge.predict(X_train)
    y_test_pred = ridge.predict(X_test)

    store_metrics(y_train, y_train_pred, y_test, y_test_pred, record, model_name)

    return ridge

def train_lasso(X_train, y_train, X_test, y_test, record, model_name='lasso'):
    param_grid = {
        'alpha': [0.01, 0.1, 1, 10, 100, 1000],
        'selection': ['cyclic', 'random'],
    }

    lasso = Lasso(random_state=12)

    grid_search = gridsearch(lasso, param_grid, X_train, y_train, record, model_name)

    lasso = Lasso(random_state=12, **grid_search.best_params_)
    lasso.fit(X_train, y_train)
    y_train_pred = lasso.predict(X_train)
    y_test_pred = lasso.predict(X_test)

    store_metrics(y_train, y_train_pred, y_test, y_test_pred, record, model_name)

    return lasso

def train_xgboost(X_train, y_train, X_test, y_test, record, model_name='xgboost'):
    param_grid = {"max_depth":    [1, 3, 5, 7, 9, 15, 25, 35, 50, 60, 70, 80, 90],
              "n_estimators": [5, 10, 20, 30, 40, 50, 70, 80, 100, 200, 300, 400, 500, 1000],
              "learning_rate": [0.01, 0.05, 0.1],
              "gamma": [0, 0.3, 0.5, 1, 5]}
    
    xgb = XGBRegressor(random_state=12)
    grid_search = gridsearch(xgb, param_grid, X_train, y_train, record, model_name)

    xgb = XGBRegressor(random_state=12, **grid_search.best_params_)
    xgb.fit(X_train, y_train)
    y_train_pred = xgb.predict(X_train)
    y_test_pred = xgb.predict(X_test)

    store_metrics(y_train, y_train_pred, y_test, y_test_pred, record, model_name)

    return xgb

def train_rf(X_train, y_train, X_test, y_test, record, model_name='rf'):
    # it takes a long time to run all the parameters. 
    # There is not need to run large ones to prevent overfitting.
    param_grid = {
        'n_estimators': [5, 10, 15],# 20], #30, 40, 60],#, 70, 80, 100],
        'max_depth': [1, 3, 5, 7, 9, 15],# 25, 35, 50, 60, 70, 80, 90],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False],
        'max_features': [0.5, 'sqrt', 'log2']
    }

    rf = RandomForestRegressor(random_state=12)
    
    grid_search = gridsearch(rf, param_grid, X_train, y_train, record, model_name)

    rf = RandomForestRegressor(random_state=12, **grid_search.best_params_)
    rf.fit(X_train, y_train)
    y_train_pred = rf.predict(X_train)
    y_test_pred = rf.predict(X_test)

    store_metrics(y_train, y_train_pred, y_test, y_test_pred, record, model_name)

    return rf

def train_svr(X_train, y_train, X_test, y_test, record, model_name='svr'):
    param_grid = {"kernel": ["rbf"], "C": np.logspace(-1, 3, 5), "gamma": np.logspace(-2, 2, 5), "epsilon": [0.1, 0.2, 0.3, 0.5]}

    svr = SVR()
    grid_search = gridsearch(svr, param_grid, X_train, y_train, record, model_name)
    svr = SVR(**grid_search.best_params_)
    svr.fit(X_train, y_train)
    y_train_pred = svr.predict(X_train)
    y_test_pred = svr.predict(X_test)

    store_metrics(y_train, y_train_pred, y_test, y_test_pred, record, model_name)

    return svr

def plot_train_test_score(record, model_name, y_train, y_test, metrics, text_pos=(5, 0)):

    y_train_pred, y_test_pred = record[model_name]['y_train_pred'], record[model_name]['y_test_pred']

    plt.scatter(y_train, y_train_pred, alpha=0.5, s=50, label='train')
    plt.scatter(y_test, y_test_pred, alpha=0.5, s=50, label ='test')
    plt.plot(np.linspace(y_train.min()-2, y_train.max()+2, 20), np.linspace(y_train.min()-2, y_train.max()+2, 20), 'r', label=r'$y=x$')


    rmse_train, rmse_test = record[model_name]['rmse_train'], record[model_name]['rmse_test']
    mape_train, mape_test = record[model_name]['mape_train'], record[model_name]['mape_test']
    r2_train, r2_test = record[model_name]['r2_train'], record[model_name]['r2_test']
    mae_train, mae_test = record[model_name]['mae_train'], record[model_name]['mae_test']

    plot_all = True if metrics == 'all' else False
    # rmse
    if plot_all or metrics == 'rmse':
        plt.text(text_pos[0], text_pos[1], r'$RMSE_{train}$' + '= {:.2f}'.format(rmse_train))
        plt.text(text_pos[0], text_pos[1]-0.7, r'$RMSE_{test}$' + '= {:.2f}'.format(rmse_test))
    # mean percentage absolute error
    if plot_all or metrics == 'mape':
        plt.text(text_pos[0], text_pos[1]-1.4, r'$MAPE_{train}$' + '= {:.2f}'.format(mape_train))
        plt.text(text_pos[0], text_pos[1]-2.1, r'$MAPE_{test}$' + '= {:.2f}'.format(mape_test))

    if plot_all or metrics == 'r2':
        plt.text(text_pos[0], text_pos[1]-2.8, r'$R2_{train}$' + '= {:.2f}'.format(r2_train))
        plt.text(text_pos[0], text_pos[1]-3.5, r'$R2_{test}$' + '= {:.2f}'.format(r2_test))

    # mae
    if plot_all or metrics == 'mae':
        plt.text(text_pos[0], text_pos[1]-4.2, r'$MAE_{train}$' + '= {:.2f}'.format(mae_train))
        plt.text(text_pos[0], text_pos[1]-4.9, r'$MAE_{test}$' + '= {:.2f}'.format(mae_test))

    plt.axis('equal')
    plt.xlabel('Truth')
    plt.ylabel('Predicted')
    plt.title(model_name)

    plt.legend()


def reg_rugplot(shap_values, feature, figsize=(6, 4), xlim=None, smooth=0):
    x = shap_values[:, feature].data
    y = shap_values[:, feature].values

    if xlim:
        inx = (x>=xlim[0]) & (x<=xlim[1])
        x = x[inx]
        y = y[inx]

    x_unique = np.unique(x)
    y_mean = [] # mean
    y_sd = [] # standard deviation
    y_se = [] # standard error, not used
    for x_ in x_unique:
        y_mean.append(np.mean(y[x==x_]))
        y_sd.append(np.std(y[x==x_]))
        y_se.append(np.std(y[x==x_])/np.sqrt(len(y[x==x_])))

    # spline
    spl = UnivariateSpline(x_unique, y_mean)
    spl.set_smoothing_factor(smooth)
    # x_new = np.linspace(x_unique.min(), x_unique.max(), 100)
    # y_new_pred = spl(x_new)
    y_pred = spl(x_unique)

    res = y_mean - y_pred
    se_pred = np.std(res)/np.sqrt(len(res))
    y_pred_low = y_pred - 1.96*se_pred
    y_pred_high = y_pred + 1.96*se_pred
    
    plt.figure(figsize=figsize)
 
    plt.plot(x_unique, spl(x_unique), alpha=0.7, color='r', label='spline')
    plt.scatter(x_unique, y_mean, alpha=0.7, label='Average SHAP value')
    plt.errorbar(x_unique, y_mean, yerr=y_sd, fmt='none', alpha=0.7)

    plt.fill_between(x_unique, y_pred_low, y_pred_high, alpha=0.3, label='95% CI')
    
    sns.rugplot(x_unique, height=0.05, alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel('SHAP value')
    plt.legend()
    plt.tight_layout()