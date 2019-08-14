#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning Demo
# ### Matthew Epland, PhD
# Partially adapted from the [sklearn documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)

# In[ ]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

########################################################
# python
import pandas as pd
import numpy as np
import warnings
from time import time
# from copy import copy
import json
from collections import OrderedDict

from scipy.io import arff
from scipy.stats import randint, uniform

########################################################
# xgboost, sklearn
import xgboost as xgb

warnings.filterwarnings('ignore', message='sklearn.externals.joblib is deprecated in 0.21 and will be removed in 0.23')
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc, roc_auc_score

########################################################
# skopt
from skopt import Optimizer
from skopt.learning import GaussianProcessRegressor, RandomForestRegressor, GradientBoostingQuantileRegressor
from skopt.learning.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.ensemble import GradientBoostingRegressor

########################################################
# hyperopt
from hyperopt import hp, tpe, fmin, Trials

########################################################
# plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

########################################################
# globals
rnd_seed = 11
n_folds = 3


# In[ ]:


from plotting import * # load plotting code from plotting.py


# In[ ]:


# Utility function to report best scores from sklearn searches
def report(results, n_top=5):
    results = results.cv_results_
    for i in range(1, n_top+1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print(f'Model with rank: {i}')
            print(f"Mean validation score: {results['mean_test_score'][candidate]:.3f} (std: {results['std_test_score'][candidate]:.3f})")
            print('Parameters: {0}\n'.format(results['params'][candidate]))


# Need to implement our own custom scorer to actually use the best number of trees found by early stopping.
# See the [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object) for details.

# In[ ]:


def xgb_early_stopping_auc_scorer(model, X, y):
    # predict_proba may not be thread safe, so copy the object - unfortunately getting crashes so just use the original object
    # model = copy.copy(model_in)
    y_pred = model.predict_proba(X, ntree_limit=model.best_ntree_limit)
    y_pred_sig = y_pred[:,1]
    return roc_auc_score(y, y_pred_sig)


# ## Load Polish Companies Bankruptcy Data
# #### [Source and data dictionary](http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)

# In[ ]:


data = arff.loadarff('./data/1year.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].apply(int, args=(2,))


# In[ ]:


# Real feature names
with open ('./attrs.json') as json_file:
    attrs_dict = json.load(json_file)


# Setup Target and Features

# In[ ]:


target='class'
features = sorted(list(set(df.columns)-set([target])))


# Make Train, Validation, and Holdout Sets

# In[ ]:


X = df[features].values
y = df[target].values

X_tmp, X_holdout, y_tmp, y_holdout = train_test_split(X, y, test_size=0.2, random_state=rnd_seed, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=rnd_seed+1, stratify=y_tmp)
del X_tmp; del y_tmp


# Prepare Stratified k-Folds

# In[ ]:


skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rnd_seed+2)


# ## Set Hyperparameter Ranges
# #### See the [docs here](https://xgboost.readthedocs.io/en/latest/parameter.html) for XGBoost hyperparameter details.

# In[ ]:


param_defaults_and_ranges = OrderedDict({
    'max_depth': {'default': 5, 'range': (3, 10), 'dist': randint(3, 10), 'grid': [4, 5, 6, 7, 8], 'hp': hp.choice('max_depth', range(3, 11))},
        # default=6, Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    'learning_rate': {'default': 0.3, 'range': (0.05, 0.6), 'dist': uniform(0.05, 0.6), 'grid': [0.05, 0.1, 0.3, 0.4], 'hp': hp.uniform('learning_rate', 0.05, 0.6)},
        # NOTE: Optimizing the log of the learning rate would be better, but avoid that complexity for this demo...
        # default=0.3, Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. alias: learning_rate
    'min_child_weight': {'default': 1., 'range': (1., 10.), 'dist': uniform(1., 10.), 'grid': [1., 2.], 'hp': hp.uniform('min_child_weight', 1., 10.)},
        # default=1, Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
    'gamma': {'default': 0., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0, 1, 2], 'hp': hp.uniform('gamma', 0., 5.)},
        # default=0, Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. alias: min_split_loss
    # 'max_delta_step': {'default': 0., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0., 1., 2.], 'hp': hp.uniform('max_delta_step', 0., 5.)},
        # default=0, Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
    # 'reg_alpha': {'default': 0., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0., 1., 3.], 'hp': hp.uniform('reg_alpha', 0., 5.)},
        # default=0, L1 regularization term on weights. Increasing this value will make model more conservative.
    # 'reg_lambda': {'default': 1., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0., 1., 3], 'hp': hp.uniform('reg_lambda', 0., 5.)},
        # default=1, L2 regularization term on weights. Increasing this value will make model more conservative.
    # 'colsample_bytree': {'default': 1., 'range': (0.5, 1.), 'dist': uniform(0.5, 1.), 'grid': [0.5, 1.], 'hp': hp.uniform('colsample_bytree', 0.5, 1.)},
        # default=1, Subsample ratio of columns when constructing each tree.
    # 'subsample': {'default': 1., 'range': (0.5, 1.), 'dist': uniform(0.5, 1.), 'grid': [0.5, 1.], 'hp': hp.uniform('subsample', 0.5, 1.)},
        # default=1, Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
})


# In[ ]:


# break out the params_to_be_opt, and their ranges (dimensions), and defaults
params_to_be_opt = []
dimensions = []
for k,v in param_defaults_and_ranges.items():
    params_to_be_opt.append(k)
    dimensions.append(v['range'])

# break out dictionaries for each optimizer
param_defaults = {}
param_dists = {}
param_grids = {}
param_hp_dists = OrderedDict()
for k,v in param_defaults_and_ranges.items():
    param_defaults[k] = v['default']
    param_dists[k] = v['dist']
    param_grids[k] = v['grid']
    param_hp_dists[k] = v['hp']

# make helper param index dict
param_index_dict = {}
for iparam, param in enumerate(params_to_be_opt):
    param_index_dict[param] = iparam


# #### Set other fixed hyperparameters

# In[ ]:


fixed_setup_params = {
'max_num_boost_rounds': 500, # maximum number of boosting rounds to run / trees to create
'xgb_objective': 'binary:logistic', # objective function for binary classification
'xgb_verbosity': 0, #  The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
'xgb_n_jobs': 1, # Number of parallel threads used to run XGBoost. -1 makes use of all cores in your system
}

# search_scoring = 'roc_auc' # need to use custom function to work properly with xgb early stopping, see xgb_early_stopping_auc_scorer
search_n_jobs = 2 # Number of parallel threads used to run hyperparameter searches
search_verbosity = 1


# In[ ]:


fixed_fit_params = {
    'early_stopping_rounds': 10, # must see improvement over last num_early_stopping_rounds or will halt
    'eval_set': [(X_val, y_val)], # data sets to use for early stopping evaluation
    'eval_metric': 'logloss', # evaluation metric for early stopping
    'verbose': False, # even more verbosity control
}


# #### Setup XGBClassifier

# In[ ]:


xgb_model = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'],
                              objective=fixed_setup_params['xgb_objective'],
                              verbosity=fixed_setup_params['xgb_verbosity'],
                              random_state=rnd_seed+3)


# #### Run with default hyperparameters as a baseline

# In[ ]:


model_default = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'],
                                  objective=fixed_setup_params['xgb_objective'],
                                  verbosity=fixed_setup_params['xgb_verbosity'],
                                  random_state=rnd_seed+3, **param_defaults)
model_default.fit(X_train, y_train, **fixed_fit_params);


# In[ ]:


y_initial = -xgb_early_stopping_auc_scorer(model_default, X_val, y_val)


# # Random Search

# In[ ]:


rs_n_iter = 10

rs = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dists, scoring=xgb_early_stopping_auc_scorer,
                        n_iter=rs_n_iter, n_jobs=search_n_jobs, cv=skf, verbose=search_verbosity, random_state=rnd_seed+4
                       )


# In[ ]:


rs_start = time()

rs.fit(X_train, y_train, groups=None, **fixed_fit_params)

rs_time = time()-rs_start

print(f'RandomizedSearchCV took {rs_time:.2f} seconds for {rs_n_iter} candidates parameter settings')


# In[ ]:


report(rs)


# In[ ]:


plot_convergence(y_values=np.array([-y for y in rs.cv_results_['mean_test_score']]), ann_text='RS', tag='_RS', y_initial=y_initial)


# In[ ]:


# search time, best model, best params, for later comparison
# print(rs_time)
# print(rs.best_estimator_)
# print(rs.best_params_)


# # Grid Search

# In[ ]:


gs = GridSearchCV(estimator=xgb_model, param_grid=param_grids, scoring=xgb_early_stopping_auc_scorer,
                  n_jobs=search_n_jobs, cv=skf, verbose=search_verbosity # , iid=False
                 )


# In[ ]:


gs_start = time()

gs.fit(X_train, y_train, groups=None, **fixed_fit_params)

gs_time = time()-gs_start

print(f"GridSearchCV took {gs_time:.2f} seconds for {len(gs.cv_results_['params'])} candidates parameter settings")


# In[ ]:


report(gs)


# In[ ]:


plot_convergence(y_values=[-y for y in gs.cv_results_['mean_test_score']], ann_text='GS', tag='_GS', y_initial=y_initial)


# In[ ]:


# search time, best model, best params, for later comparison
# print(gs_time)
# print(gs.best_estimator_)
# print(gs.best_params_)


# # Setup datasets and objective function for custom searches

# In[ ]:


# make _BO train and val sets from regular train set, will use these for early stopping and regular val set for testing while iterating in the optimizers
X_train_OPT, X_val_OPT, y_train_OPT, y_val_OPT = train_test_split(X_train, y_train, test_size=0.2, random_state=rnd_seed+5, stratify=y_train)


# In[ ]:


# setup the function to be optimized
def objective_function(params):
    model = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'], objective=fixed_setup_params['xgb_objective'], verbosity=fixed_setup_params['xgb_verbosity'], random_state=rnd_seed+6, **params)
    model.fit(X_train_OPT, y_train_OPT, early_stopping_rounds=fixed_fit_params['early_stopping_rounds'], eval_set=[(X_val_OPT, y_val_OPT)], eval_metric=fixed_fit_params['eval_metric'], verbose=fixed_fit_params['verbose'])

    best_ntree_limit = model.best_ntree_limit
    if best_ntree_limit >= fixed_setup_params['max_num_boost_rounds']:
        print(f"Hit max_num_boost_rounds = {fixed_setup_params['max_num_boost_rounds']:d}, model.best_ntree_limit = {best_ntree_limit:d}")

    # return the negative auc of the trained model, since Optimizer and hyperopt only minimize
    return -xgb_early_stopping_auc_scorer(model, X_val, y_val)


# # Bayesian Optimization

# In[ ]:


n_initial_points = 20
acq_func='gp_hedge' # select the best of EI, PI, LCB per iteration


# In[ ]:


def run_bo(bo_opt, bo_n_iter, ann_text, m_path='output', tag='', params_initial=None, y_initial=None, print_interval=5):
    iter_results = []
    if params_initial is not None and y_initial is not None:
        # update bo_opt with the initial / default point, might as well since we have already computed it!
        x_initial = [params_initial[param] for param in params_to_be_opt]
        bo_opt.tell(x_initial, y_initial)

        iter_results.append({'iter':0, 'y':y_initial, 'auc':-y_initial, 'x': str(x_initial)})

    # we'll print these warnings ourselves
    warnings.filterwarnings('ignore', message='The objective has been evaluated at this point before.')

    # run it
    for i_iter in range(1,bo_n_iter):
        is_last = False
        if i_iter+1 == bo_n_iter:
            is_last = True

        print_this_i = False
        if is_last or (print_interval !=0 and (print_interval < 0 or (print_interval > 0 and i_iter % print_interval == 0))):
            print_this_i = True
            print(f'Starting iteration {i_iter:d}')

        # get next test point x, ie a new point beta in parameter space
        x = bo_opt.ask()

        is_repeat = False
        if x in bo_opt.Xi:
            is_repeat = True
            # we have already evaluated objective_function at this point! Pull old value, give it back and continue.
            # not very elegant, might still get a warning from Optimizer, but at least is MUCH faster than recomputing objective_function...
            past_i_iter = bo_opt.Xi.index(x)
            y = bo_opt.yi[past_i_iter] # get from bo_opt array to be sure it's the right one

            print('Already evaluated at this x (params below)! Just using the old result for y and continuing!')
            print(x)

        else:
            # run the training and predictions for the test point
            params = {}
            for param,value in zip(params_to_be_opt,x):
                params[param] = value
            y = objective_function(params)
        # update bo_opt with the result for the test point
        bo_opt.tell(x, y)

        # save to df
        iter_result = {'iter':i_iter, 'y':y, 'auc':-y, 'x': str(x)}
        for param,value in zip(params_to_be_opt,x):
            iter_result[param] = value
        iter_results.append(iter_result)

        # see if it is a min
        is_best = False
        if i_iter != 0 and y == np.array(bo_opt.yi).min():
            is_best = True

        # print messages and plots while still running
        if print_this_i or is_last or (is_best and not is_repeat):
            if is_best:
                print('Found a new optimum set of hyper parameters!')
            print(x)
            print(f'y: {y:.5f}')

            df_tmp = pd.DataFrame.from_dict(iter_results)

            plot_convergence(df_tmp['y'], ann_text, m_path, tag=tag, y_initial=y_initial)

    # print best values
    yi = np.array(bo_opt.yi)
    y_best = yi.min()
    x_best_indices = np.where(yi == y_best)
    x_best_indices = x_best_indices[0]

    best_params_points_list = []
    for this_x_best_index in x_best_indices:
        this_x_best = bo_opt.Xi[this_x_best_index]

        this_x_best_params = {}
        this_x_best_params['iter'] = this_x_best_index
        for param in params_to_be_opt:
            this_x_best_params[param] = this_x_best[param_index_dict[param]]

        best_params_points_list.append(this_x_best_params)

    # save best results to csv
    df_best_params_points = pd.DataFrame.from_dict(best_params_points_list)
    df_best_params_points = df_best_params_points.drop_duplicates(subset=params_to_be_opt).reset_index(drop=True)
    df_best_params_points = df_best_params_points.sort_values(by=params_to_be_opt).reset_index(drop=True)
    df_best_params_points = df_best_params_points[['iter']+params_to_be_opt]

    df_best_params_points.to_csv(f'{m_path}/best_params_points{tag}.csv', index=False, na_rep='nan')

    # save iter results to csv
    df_iter_results = pd.DataFrame.from_dict(iter_results)
    fixed_cols = ['iter', 'y', 'auc']
    cols = fixed_cols + list(set(df_iter_results.columns)-set(fixed_cols))
    df_iter_results = df_iter_results[cols]
    df_iter_results = df_iter_results.sort_values(by='iter').reset_index(drop=True)
    df_iter_results.to_csv(f'{m_path}/iter_results{tag}.csv', index=False, na_rep='nan')

    n_best_params_points = len(df_best_params_points)

    print('Best hyperparameters:')
    for index,row in df_best_params_points.iterrows():
        if n_best_params_points > 1:
            print(f'\nPoint {index}:')

        for param in params_to_be_opt:
            print(f'{param}: {row[param]}')

    if n_best_params_points > 1:
        print('Parameter Mean and St Dev:')
        for param in params_to_be_opt:
            param_values = df_best_params_points[param].values
            print(f'{param} Mean: {np.mean(param_values)}, St Dev: {np.std(param_values)}')

    if y_initial is not None:
        print(f'   Best y: {y_best:.5f} (should be smaller than initial), a decrease of {y_initial-y_best:.5f}, {(y_initial-y_best)/y_initial:.3%}')
        print(f'Initial y: {y_initial:.5f}')


# ### Gaussian Process Surrogate

# In[ ]:


# radial basis function + white noise kernel
bo_gp_opt = Optimizer(dimensions=dimensions, n_initial_points=n_initial_points, acq_func=acq_func, random_state=rnd_seed+6,
                      base_estimator=GaussianProcessRegressor(
                          kernel=RBF(length_scale_bounds=[1.0e-3, 1.0e+3]) + WhiteKernel(noise_level=1.0e-5, noise_level_bounds=[1.0e-6, 1.0e-2])
                      ),
                     )


# In[ ]:


run_bo(bo_gp_opt, bo_n_iter=50, ann_text='GP', tag='_GP', params_initial=param_defaults, y_initial=y_initial, print_interval=25)


# ### Random Forest Surrogate

# In[ ]:


bo_rf_opt = Optimizer(dimensions=dimensions, n_initial_points=n_initial_points, acq_func=acq_func, random_state=rnd_seed+7,
                      base_estimator=RandomForestRegressor(n_estimators=200, max_depth=8, random_state=rnd_seed+8),
                     )


# In[ ]:


run_bo(bo_rf_opt, bo_n_iter=50, ann_text='RF', tag='_RF', params_initial=param_defaults, y_initial=y_initial, print_interval=25)


# ### Gradient Boosted Trees Surrogate

# In[ ]:


gbrt_base_estimator = GradientBoostingQuantileRegressor(
    base_estimator=GradientBoostingRegressor(loss='quantile', max_depth=8, learning_rate=0.1, n_estimators=200,
                                             n_iter_no_change=10, validation_fraction=0.2, tol=0.0001, random_state=rnd_seed+9)
)

bo_bdt_opt = Optimizer(dimensions=dimensions, n_initial_points=n_initial_points, acq_func=acq_func, random_state=rnd_seed+10, base_estimator=gbrt_base_estimator)


# In[ ]:


run_bo(bo_bdt_opt, bo_n_iter=50, ann_text='GBDT', tag='_GBDT', params_initial=param_defaults, y_initial=y_initial, print_interval=25)


# # Tree-Structured Parzen Estimator (TPE)
# Note that hyperopt with TPE can accommodate nested hyperparameter search distributions. See [here](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a#951b) for more.

# In[ ]:


tpe_trials = Trials()

tpe_best = fmin(fn=objective_function, space=param_hp_dists, algo=tpe.suggest, max_evals=20, trials=tpe_trials)


# In[ ]:


plot_convergence(y_values=[result['loss'] for result in tpe_trials.results], ann_text='TPE', tag='_TPE', y_initial=y_initial)


# # Genetic Algorithm TODO

# # Evaluate Performance TODO

# In[ ]:


my_plot_evaluations((rs, param_hp_dists), ann_text='RS', tag='_RS', bins=10, dimensions=params_to_be_opt)


# In[ ]:


my_plot_evaluations((gs, param_hp_dists), ann_text='GS', tag='_GS', bins=10, dimensions=params_to_be_opt)


# In[ ]:


my_plot_evaluations(bo_gp_opt, ann_text='GP', tag='_GP', bins=10, dimensions=params_to_be_opt)
my_plot_objective(bo_gp_opt, ann_text='GP', tag='_GP', dimensions=params_to_be_opt)


# In[ ]:


my_plot_evaluations(bo_rf_opt, ann_text='RF', tag='_RF', bins=10, dimensions=params_to_be_opt)
my_plot_objective(bo_rf_opt, ann_text='RF', tag='_RF', dimensions=params_to_be_opt) # takes much longer for these partial dependencies


# In[ ]:


my_plot_evaluations(bo_bdt_opt, ann_text='GBDT', tag='_GBDT', bins=10, dimensions=params_to_be_opt)
my_plot_objective(bo_bdt_opt, ann_text='GBDT', tag='_GBDT', dimensions=params_to_be_opt)


# In[ ]:


my_plot_evaluations((tpe_trials, param_hp_dists), ann_text='TPE', tag='_TPE', bins=10, dimensions=params_to_be_opt)


# In[ ]:





# # Dev

# In[ ]:


from plotting import *


# In[ ]:


y_holdout_pred = model.predict_proba(X_holdout, ntree_limit=model.best_ntree_limit)[:,1]


# In[ ]:


plot_y_pred(y_holdout_pred, y_holdout)


# In[ ]:


fpr, tpr, thr = roc_curve(y_holdout, y_holdout_pred)


# In[ ]:


plot_roc(fpr, tpr)


# In[ ]:




