#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning Demo
# ### Matthew Epland, PhD
# Adapted from:
# * [Sklearn Documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)
# * [yandexdataschool/mlhep2018 Slides](https://github.com/yandexdataschool/mlhep2018/blob/master/day4-Fri/Black-Box.pdf)
# * [Hyperparameter Optimization in Python Part 1: Scikit-Optimize](https://towardsdatascience.com/hyperparameter-optimization-in-python-part-1-scikit-optimize-754e485d24fe)
# * [An Introductory Example of Bayesian Optimization in Python with Hyperopt](https://towardsdatascience.com/an-introductory-example-of-bayesian-optimization-in-python-with-hyperopt-aae40fff4ff0)

# Install required packages via pip if necessary, only run if you know what you're doing! [Reference](https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/)  
# **Note: This does not use a virtual environment and will pip install directly to your system!**

# In[ ]:


import sys
get_ipython().system('{sys.executable} -m pip install --upgrade pip')
get_ipython().system('{sys.executable} -m pip install -r requirements.txt')
# !{sys.executable} -m pip install -r gentun/requirements.txt


# In[ ]:


# import sys
# !{sys.executable} -m pip uninstall --yes gentun


# In[ ]:


# %%bash
# cd gentun/
# python3 setup.py install


# Check how many cores we have

# In[4]:


import multiprocessing
multiprocessing.cpu_count()


# ### Load packages!

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

########################################################
# python
import pandas as pd
import numpy as np
import warnings
from time import time
# from copy import copy
from collections import OrderedDict
import json
import pickle

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
# set global rnd_seed for reproducibility
rnd_seed = 42


# In[2]:


from utils import * # load some helper functions, but keep main body of code in notebook for easier reading


# In[3]:


from plotting import * # load plotting code


# ### Set number of iterations

# In[4]:


n_iters = {
    'RS': 500,
     # 'GS': set by the size of the grid
    'GP': 300,
    'RF': 300,
    'GBDT': 300,
    'TPE': 300,
    # 'GA': 300, # number of generations
}

# all will effectively be multiplied by n_folds
n_folds = 5

# for testing lower iterations and folds
for k,v in n_iters.items():
    n_iters[k] = 30

# n_iters['GA'] = 1
n_folds = 2
# Need to implement our own custom scorer to actually use the best number of trees found by early stopping.
# See the [documentation](https://scikit-learn.org/stable/modules/model_evaluation.html#implementing-your-own-scoring-object) for details.

# In[5]:


def xgb_early_stopping_auc_scorer(model, X, y):
    # predict_proba may not be thread safe, so copy the object - unfortunately getting crashes so just use the original object
    # model = copy.copy(model_in)
    y_pred = model.predict_proba(X, ntree_limit=model.best_ntree_limit)
    y_pred_sig = y_pred[:,1]
    return roc_auc_score(y, y_pred_sig)


# ## Load Polish Companies Bankruptcy Data
# ### [Source and data dictionary](http://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data)

# In[6]:


data = arff.loadarff('./data/1year.arff')
df = pd.DataFrame(data[0])
df['class'] = df['class'].apply(int, args=(2,))


# In[7]:


# Real feature names, for reference
with open ('./attrs.json') as json_file:
    attrs_dict = json.load(json_file)


# Setup Target and Features

# In[8]:


target='class'
features = sorted(list(set(df.columns)-set([target])))


# Make Train, Validation, and Holdout Sets

# In[9]:


X = df[features].values
y = df[target].values

X_trainCV, X_holdout, y_trainCV, y_holdout = train_test_split(X, y, test_size=0.2, random_state=rnd_seed, stratify=y)
del X; del y;

dm_train = xgb.DMatrix(X_trainCV, label=y_trainCV)

X_train, X_val, y_train, y_val = train_test_split(X_trainCV, y_trainCV, test_size=0.2, random_state=rnd_seed, stratify=y_trainCV)


# Prepare Stratified k-Folds

# In[10]:


skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rnd_seed+2)


# ## Setup Hyperparameter Search Space
# See the [docs here](https://xgboost.readthedocs.io/en/latest/parameter.html) for XGBoost hyperparameter details.

# In[11]:


all_params = OrderedDict({
    'max_depth': {'initial': 6, 'range': (3, 10), 'dist': randint(3, 10), 'grid': [4, 6, 8], 'hp': hp.choice('max_depth', [3,4,5,6,7,8,9,10])},
        # default=6, Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    'learning_rate': {'initial': 0.3, 'range': (0.05, 0.6), 'dist': uniform(0.05, 0.6), 'grid': [0.1, 0.15, 0.3], 'hp': hp.uniform('learning_rate', 0.05, 0.6)},
        # NOTE: Optimizing the log of the learning rate would be better, but avoid that complexity for this demo...
        # default=0.3, Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. alias: learning_rate
    # 'min_child_weight': {'initial': 1., 'range': (1., 10.), 'dist': uniform(1., 10.), 'grid': [1., 3.], 'hp': hp.uniform('min_child_weight', 1., 10.)},
        # default=1, Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
    'gamma': {'initial': 0., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0., 0.5, 1.], 'hp': hp.uniform('gamma', 0., 5.)},
        # default=0, Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. alias: min_split_loss
    'reg_alpha': {'initial': 0., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0., 1.], 'hp': hp.uniform('reg_alpha', 0., 5.)},
        # default=0, L1 regularization term on weights. Increasing this value will make model more conservative.
    'reg_lambda': {'initial': 1., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0., 1.], 'hp': hp.uniform('reg_lambda', 0., 5.)},
        # default=1, L2 regularization term on weights. Increasing this value will make model more conservative.
    # 'max_delta_step': {'initial': 0., 'range': (0., 5.), 'dist': uniform(0., 5.), 'grid': [0., 1.], 'hp': hp.uniform('max_delta_step', 0., 5.)},
        # default=0, Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
    # TODO debug ranges (0, 1) so they are actually working
    # 'colsample_bytree': {'initial': 1., 'range': (0.5, 1.), 'dist': uniform(0.5, 1.), 'grid': [0.5, 1.], 'hp': hp.uniform('colsample_bytree', 0.5, 1.)},
        # default=1, Subsample ratio of columns when constructing each tree.
    # 'subsample': {'initial': 1., 'range': (0.5, 1.), 'dist': uniform(0.5, 1.), 'grid': [0.5, 1.], 'hp': hp.uniform('subsample', 0.5, 1.)},
        # default=1, Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
})


# In[12]:


# break out the params_to_be_opt, and their ranges (dimensions), and initial values
params_to_be_opt = []
dimensions = []
for k,v in all_params.items():
    params_to_be_opt.append(k)
    dimensions.append(v['range'])

# break out dictionaries for each optimizer
params_initial = {}
param_dists = {}
param_grids = {}
param_hp_dists = OrderedDict()
for k,v in all_params.items():
    params_initial[k] = v['initial']
    param_dists[k] = v['dist']
    param_grids[k] = v['grid']
    param_hp_dists[k] = v['hp']

# make helper param index dict
param_index_dict = {}
for iparam, param in enumerate(params_to_be_opt):
    param_index_dict[param] = iparam


# #### Set other fixed hyperparameters

# In[13]:


fixed_setup_params = {
'max_num_boost_rounds': 500, # maximum number of boosting rounds to run / trees to create
'xgb_objective': 'binary:logistic', # objective function for binary classification
'xgb_verbosity': 0, #  The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
'xgb_n_jobs': -1, # Number of parallel threads used to run XGBoost. -1 makes use of all cores in your system
}

# search_scoring = 'roc_auc' # need to use custom function to work properly with xgb early stopping, see xgb_early_stopping_auc_scorer
search_n_jobs = -1 # Number of parallel threads used to run hyperparameter searches. -1 makes use of all cores in your system
search_verbosity = 1


# In[14]:


fixed_fit_params = {
    'early_stopping_rounds': 10, # must see improvement over last num_early_stopping_rounds or will halt
    'eval_set': [(X_val, y_val)], # data sets to use for early stopping evaluation
    'eval_metric': 'auc', # evaluation metric for early stopping
    'verbose': False, # even more verbosity control
}


# ### Setup XGBClassifier

# In[15]:


xgb_model = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'],
                              objective=fixed_setup_params['xgb_objective'],
                              verbosity=fixed_setup_params['xgb_verbosity'],
                              random_state=rnd_seed+3)


# #### Run with initial hyperparameters as a baseline

# In[16]:


model_initial = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'],
                                  objective=fixed_setup_params['xgb_objective'],
                                  verbosity=fixed_setup_params['xgb_verbosity'],
                                  random_state=rnd_seed+3, **params_initial)
model_initial.fit(X_train, y_train, **fixed_fit_params);


# In[17]:


y_initial = -xgb_early_stopping_auc_scorer(model_initial, X_val, y_val)


# # Random Search
# Randomly test different hyperparameters drawn from `param_dists`

# In[22]:


rs = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_dists, scoring=xgb_early_stopping_auc_scorer,
                        n_iter=n_iters['RS'], n_jobs=search_n_jobs, cv=skf, verbose=search_verbosity, random_state=rnd_seed+4
                       )


# In[23]:


rs_start = time()

rs.fit(X_trainCV, y_trainCV, groups=None, **fixed_fit_params)
dump_to_pkl(rs, 'RS')

rs_time = time()-rs_start

print(f"RandomizedSearchCV took {rs_time:.2f} seconds for {n_iters['RS']} candidates parameter settings")


# In[34]:


rs = load_from_pkl('RS')


# In[25]:


report(rs)


# In[26]:


output_sklearn_to_csv(rs, params_to_be_opt, tag='_RS')


# In[27]:


plot_convergence(y_values=np.array([-y for y in rs.cv_results_['mean_test_score']]), ann_text='RS', tag='_RS', y_initial=y_initial)


# # Grid Search
# Try all possible hyperparameter combinations `param_grids`, slow and poor exploration!

# In[28]:


gs = GridSearchCV(estimator=xgb_model, param_grid=param_grids, scoring=xgb_early_stopping_auc_scorer,
                  n_jobs=search_n_jobs, cv=skf, verbose=search_verbosity # , iid=False
                 )


# In[29]:


gs_start = time()

gs.fit(X_trainCV, y_trainCV, groups=None, **fixed_fit_params)
dump_to_pkl(gs, 'GS')

gs_time = time()-gs_start

print(f"GridSearchCV took {gs_time:.2f} seconds for {len(gs.cv_results_['params'])} candidates parameter settings")


# In[35]:


gs = load_from_pkl('GS')


# In[31]:


report(gs)


# In[32]:


output_sklearn_to_csv(gs, params_to_be_opt, tag='_GS')


# In[33]:


plot_convergence(y_values=[-y for y in gs.cv_results_['mean_test_score']], ann_text='GS', tag='_GS', y_initial=y_initial)


# # Setup datasets and objective function for custom searches
# setup the function to be optimized - without CV
def objective_function(params):
    model = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'], objective=fixed_setup_params['xgb_objective'], verbosity=fixed_setup_params['xgb_verbosity'], random_state=rnd_seed+6, **params)
    model.fit(X_train_OPT, y_train_OPT, early_stopping_rounds=fixed_fit_params['early_stopping_rounds'], eval_set=[(X_val_OPT, y_val_OPT)], eval_metric=fixed_fit_params['eval_metric'], verbose=fixed_fit_params['verbose'])

    best_ntree_limit = model.best_ntree_limit
    if best_ntree_limit >= fixed_setup_params['max_num_boost_rounds']:
        print(f"Hit max_num_boost_rounds = {fixed_setup_params['max_num_boost_rounds']:d}, model.best_ntree_limit = {best_ntree_limit:d}")

    # return the negative auc of the trained model, since Optimizer and hyperopt only minimize
    return -xgb_early_stopping_auc_scorer(model, X_val, y_val)
# In[19]:


# setup the function to be optimized
def objective_function(params):
    cv = xgb.cv(dict({'objective': fixed_setup_params['xgb_objective']}, **params), dm_train,
                num_boost_round=fixed_setup_params['max_num_boost_rounds'], early_stopping_rounds=fixed_fit_params['early_stopping_rounds'],
                nfold=n_folds, stratified=True, folds=skf,
                metrics=fixed_fit_params['eval_metric'],
                verbose_eval=fixed_fit_params['verbose'], seed=rnd_seed+6, as_pandas=True)

    # return the negative auc of the trained model, since Optimizer and hyperopt only minimize
    return -cv[f"test-{fixed_fit_params['eval_metric']}-mean"].iloc[-1]


# # Bayesian Optimization
# Use Bayesian optimization to intelligently decide where to sample the objective function next, based on prior results.  
# Can use many different types of surrogate functions: Gaussian Process, Random Forest, Gradient Boosted Trees. Note that the Tree-Structured Parzen Estimator (TPE) approach is a close cousin of Bayesian optimization, similar in operation but arising from a flipped form of Bayes rule.

# In[20]:


frac_initial_points = 0.1
acq_func='gp_hedge' # select the best of EI, PI, LCB per iteration


# In[21]:


def run_bo(bo_opt, bo_n_iter, ann_text, m_path='output', tag='', params_initial=None, y_initial=None, print_interval=5, debug=False):
    iter_results = []
    if params_initial is not None and y_initial is not None:
        # update bo_opt with the initial point, might as well since we have already computed it!
        x_initial = [params_initial[param] for param in params_to_be_opt]
        bo_opt.tell(x_initial, y_initial)

        initial_row = {'iter':0, 'y':y_initial, 'auc':-y_initial}
        for param in params_to_be_opt:
            initial_row[param] = params_initial[param]
        iter_results.append(initial_row)

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

            if debug:
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
        iter_result = {'iter':i_iter, 'y':y, 'auc':-y} # , 'x': str(x)
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

    # save iter results to csv
    df_iter_results = pd.DataFrame.from_dict(iter_results)
    df_iter_results = df_iter_results.sort_values(by='iter').reset_index(drop=True)
    fixed_cols = ['iter', 'y', 'auc']+params_to_be_opt
    cols = fixed_cols + list(set(df_iter_results.columns)-set(fixed_cols))
    df_iter_results = df_iter_results[cols]
    df_iter_results.to_csv(f'{m_path}/iter_results{tag}.csv', index=False, na_rep='nan')

    # save best results to csv
    y_best = df_iter_results['y'].min()
    df_best = df_iter_results.loc[y_best == df_iter_results['y']]
    df_best = df_best.drop_duplicates(subset=params_to_be_opt).reset_index(drop=True)
    df_best = df_best.sort_values(by=params_to_be_opt).reset_index(drop=True)
    df_best = df_best[['y', 'auc']+params_to_be_opt]
    df_best.to_csv(f'{m_path}/best_params_points{tag}.csv', index=False, na_rep='nan')

    n_best_params_points = len(df_best)

    print('\nBest hyperparameters:')
    for index,row in df_best.iterrows():
        if n_best_params_points > 1:
            print(f'\nPoint {index}:')

        for param in params_to_be_opt:
            print(f'{param}: {row[param]}')

    if n_best_params_points > 1:
        print('Parameter Mean and St Dev:')
        for param in params_to_be_opt:
            param_values = df_best[param].values
            print(f'{param} Mean: {np.mean(param_values)}, St Dev: {np.std(param_values)}')

    if y_initial is not None:
        print(f'\n   Best y: {y_best:.5f} (should be smaller than initial), a decrease of {y_initial-y_best:.5f}, {(y_initial-y_best)/y_initial:.3%}')
        print(f'Initial y: {y_initial:.5f}')


# ## Gaussian Process Surrogate

# In[37]:


# radial basis function + white noise kernel
bo_gp_opt = Optimizer(dimensions=dimensions, n_initial_points=np.ceil(frac_initial_points*n_iters['GP']), acq_func=acq_func, random_state=rnd_seed+6,
                      base_estimator=GaussianProcessRegressor(
                          kernel=RBF(length_scale_bounds=[1.0e-3, 1.0e+3]) + WhiteKernel(noise_level=1.0e-5, noise_level_bounds=[1.0e-6, 1.0e-2])
                      ),
                     )


# In[38]:


run_bo(bo_gp_opt, bo_n_iter=n_iters['GP'], ann_text='GP', tag='_GP', params_initial=params_initial, y_initial=y_initial, print_interval=50)
dump_to_pkl(bo_gp_opt, 'GP')


# In[22]:


bo_gp_opt = load_from_pkl('GP')


# ## Random Forest Surrogate

# In[23]:


bo_rf_opt = Optimizer(dimensions=dimensions, n_initial_points=np.ceil(frac_initial_points*n_iters['RF']), acq_func=acq_func, random_state=rnd_seed+7,
                      base_estimator=RandomForestRegressor(n_estimators=200, max_depth=8, random_state=rnd_seed+8),
                     )


# In[24]:


run_bo(bo_rf_opt, bo_n_iter=n_iters['RF'], ann_text='RF', tag='_RF', params_initial=params_initial, y_initial=y_initial, print_interval=50)
dump_to_pkl(bo_rf_opt, 'RF')


# In[25]:


bo_rf_opt = load_from_pkl('RF')


# ## Gradient Boosted Trees Surrogate

# In[26]:


gbrt_base_estimator = GradientBoostingQuantileRegressor(
    base_estimator=GradientBoostingRegressor(loss='quantile', max_depth=8, learning_rate=0.1, n_estimators=200,
                                             n_iter_no_change=10, validation_fraction=0.2, tol=0.0001, random_state=rnd_seed+9)
)

bo_gbdt_opt = Optimizer(dimensions=dimensions, n_initial_points=np.ceil(frac_initial_points*n_iters['GBDT']), acq_func=acq_func,
                        random_state=rnd_seed+10, base_estimator=gbrt_base_estimator)


# In[27]:


run_bo(bo_gbdt_opt, bo_n_iter=n_iters['GBDT'], ann_text='GBDT', tag='_GBDT', params_initial=params_initial, y_initial=y_initial, print_interval=50)
dump_to_pkl(bo_gbdt_opt, 'GBDT')


# In[28]:


bo_gbdt_opt = load_from_pkl('GBDT')


# # Tree-Structured Parzen Estimator (TPE)
# Note that hyperopt with TPE can accommodate nested hyperparameter search distributions. See [here](https://towardsdatascience.com/automated-machine-learning-hyperparameter-tuning-in-python-dfda59b72f8a#951b) for an example.

# In[29]:


tpe_trials = Trials()

tpe_best = fmin(fn=objective_function, space=param_hp_dists, algo=tpe.suggest, max_evals=n_iters['TPE'],
                trials=tpe_trials, rstate=np.random.RandomState(rnd_seed+11))
dump_to_pkl(tpe_trials, 'TPE')


# In[30]:


tpe_trials = load_from_pkl('TPE')


# In[31]:


plot_convergence(y_values=tpe_trials.losses(), ann_text='TPE', tag='_TPE', y_initial=y_initial)


# In[32]:


output_hyperopt_to_csv(tpe_trials, params_to_be_opt, tag='_TPE')


# # Genetic Algorithm

# #### Eventual TODOs
# * Multithreading
# * Set random seed, but would require a careful rewrite of gentun

# In[23]:


from gentun import GeneticAlgorithm, Population, GridPopulation, XgboostIndividual

# Generate a grid of individuals as the initial population
# Use the same grid as in the sklearn grid search, and the first generation will be the same as that grid search
# large computation, requires working multithreading / distributed computing
pop = GridPopulation(XgboostIndividual, X_trainCV, y_trainCV, genes_grid=param_grids,
                     additional_parameters={'kfold': n_folds,
                                            'objective': fixed_setup_params['xgb_objective'],
                                            'eval_metric': fixed_fit_params['eval_metric'],
                                            'num_boost_round': fixed_setup_params['max_num_boost_rounds'],
                                            'early_stopping_rounds': fixed_fit_params['early_stopping_rounds'],
                                            'folds': skf, # stratified kfolds from sklearn
                                            'verbose_eval': fixed_fit_params['verbose'],
                                           },
                     crossover_rate=0.5, mutation_rate=0.02, maximize=True)
# In[33]:


# proof of concept code, turn off kfolds, run a low number of iterations
# Generate a random sample of 25 individuals as the initial population
pop = Population(XgboostIndividual, X_trainCV, y_trainCV,
                 size=25,
                 additional_parameters={# 'kfold': n_folds,
                                        'objective': fixed_setup_params['xgb_objective'],
                                        'eval_metric': fixed_fit_params['eval_metric'],
                                        'num_boost_round': fixed_setup_params['max_num_boost_rounds'],
                                        'early_stopping_rounds': fixed_fit_params['early_stopping_rounds'],
                                        # 'folds': skf, # stratified kfolds from sklearn
                                        'verbose_eval': fixed_fit_params['verbose'],
                                       },
                 crossover_rate=0.5, mutation_rate=0.05, maximize=True)


# In[34]:


ga = GeneticAlgorithm(pop, tournament_size=5, elitism=True, verbosity=1)


# In[35]:


n_iters['GA'] = 10


# In[36]:


ga.run(n_iters['GA'])


# In[37]:


ga_results = ga.get_results()
ga_results = ga_results[['generation', 'best_fitness']+params_to_be_opt]
dump_to_pkl(ga_results, 'GA') # is just a df, but might as well still pkl to be consistent


# In[38]:


ga_results = load_from_pkl('GA')


# In[40]:


plot_convergence(y_values=np.array([-y for y in ga_results['best_fitness'].to_list()]), ann_text='GA', tag='_GA', y_initial=y_initial)


# In[41]:


output_gentun_to_csv(ga_results, params_to_be_opt, tag='_GA')


# # Evaluate Performance
# ### Make evaluation and objective (when possible) plots from skopt

# In[36]:


my_plot_evaluations((rs, param_hp_dists), ann_text='RS', tag='_RS', bins=10, dimensions=params_to_be_opt)


# In[37]:


my_plot_evaluations((gs, param_hp_dists), ann_text='GS', tag='_GS', bins=10, dimensions=params_to_be_opt)


# In[38]:


my_plot_evaluations(bo_gp_opt, ann_text='GP', tag='_GP', bins=10, dimensions=params_to_be_opt)
my_plot_objective(bo_gp_opt, ann_text='GP', tag='_GP', dimensions=params_to_be_opt)


# In[39]:


my_plot_evaluations(bo_rf_opt, ann_text='RF', tag='_RF', bins=10, dimensions=params_to_be_opt)
my_plot_objective(bo_rf_opt, ann_text='RF', tag='_RF', dimensions=params_to_be_opt) # takes much longer for these partial dependencies


# In[40]:


my_plot_evaluations(bo_gbdt_opt, ann_text='GBDT', tag='_GBDT', bins=10, dimensions=params_to_be_opt)
my_plot_objective(bo_gbdt_opt, ann_text='GBDT', tag='_GBDT', dimensions=params_to_be_opt)


# In[41]:


my_plot_evaluations((tpe_trials, param_hp_dists), ann_text='TPE', tag='_TPE', bins=10, dimensions=params_to_be_opt)


# In[39]:


my_plot_evaluations((ga_results, param_hp_dists), ann_text='GA', tag='_GA', bins=10, dimensions=params_to_be_opt)


# ### Load best parameters from all optimizers

# In[42]:


optimizer_abbrevs = ['RS', 'GS', 'GP', 'RF', 'GBDT', 'TPE', 'GA']

df_best_results = combine_best_results(optimizer_abbrevs, params_to_be_opt, params_initial, y_initial, m_path='output')


# In[43]:


df_best_results


# In[44]:


df_best_results.to_csv('./output/best_results_all.csv', index=False, na_rep='nan')


# ### Evaluate models with the best parameters from each optimizer

# In[45]:


def eval_best_models(df_best_results):
    best_models = {}
    for index, row in df_best_results.iterrows():
        params = {param: row[param] for k in params_to_be_opt}

        this_model = xgb.XGBClassifier(n_estimators=fixed_setup_params['max_num_boost_rounds'],
                                       objective=fixed_setup_params['xgb_objective'],
                                       verbosity=fixed_setup_params['xgb_verbosity'],
                                       random_state=rnd_seed+12, **params)
        this_model.fit(X_train, y_train, **fixed_fit_params)

        y_holdout_pred = this_model.predict_proba(X_holdout, ntree_limit=this_model.best_ntree_limit)[:,1]

        fpr, tpr, thr = roc_curve(y_holdout, y_holdout_pred)

        best_models[row['optimizer']] = dict({'model': this_model, 'y_holdout_pred': y_holdout_pred, 'fpr': fpr, 'tpr': tpr, 'thr': thr}, **row)

    return best_models


# In[46]:


best_models = eval_best_models(df_best_results)


# ### Plot ROC curves

# In[47]:


models_for_roc= [
    {'name': 'Initial', 'nname': 'Initial', 'fpr': best_models['Initial']['fpr'], 'tpr': best_models['Initial']['tpr'], 'c': 'black', 'ls': '-'},
    {'name': 'RS', 'nname': 'RS', 'fpr': best_models['RS']['fpr'], 'tpr': best_models['RS']['tpr'], 'c': 'C0', 'ls': ':'},
    {'name': 'GS', 'nname': 'GS', 'fpr': best_models['GS']['fpr'], 'tpr': best_models['GS']['tpr'], 'c': 'C1', 'ls': '-.'},
    {'name': 'GP', 'nname': 'GP', 'fpr': best_models['GP']['fpr'], 'tpr': best_models['GP']['tpr'], 'c': 'C2', 'ls': '-'},
    {'name': 'RF', 'nname': 'RF', 'fpr': best_models['RF']['fpr'], 'tpr': best_models['RF']['tpr'], 'c': 'C3', 'ls': ':'},
    {'name': 'GBDT', 'nname': 'GBDT', 'fpr': best_models['GBDT']['fpr'], 'tpr': best_models['GBDT']['tpr'], 'c': 'C4', 'ls': '--'},
    {'name': 'TPE', 'nname': 'TPE', 'fpr': best_models['TPE']['fpr'], 'tpr': best_models['TPE']['tpr'], 'c': 'C5', 'ls': '-.'},
    {'name': 'GA', 'nname': 'GA', 'fpr': best_models['GA']['fpr'], 'tpr': best_models['GA']['tpr'], 'c': 'C6', 'ls': '--'},
]


# In[48]:


plot_rocs(models_for_roc, rndGuess=True, inverse_log=False, inline=False)
plot_rocs(models_for_roc, rndGuess=False, inverse_log=True, tag='_inverse_log', x_axis_params={'max':0.4}, y_axis_params={'max':1e1}, inline=False)


# ### Plot $\hat{y}$ predictions

# In[49]:


for k,v in best_models.items():
    plot_y_pred(v['y_holdout_pred'], y_holdout, tag=f'_{k}', ann_text=k, nbins=20)


# # Dev

# In[ ]:


raise ValueError('Stop Here, in Dev!')


# In[ ]:


from plotting import *


# In[ ]:




