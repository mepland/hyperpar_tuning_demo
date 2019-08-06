#!/usr/bin/env python
# coding: utf-8

# # Hyperparameter Tuning Demo
# ### Matthew Epland
# Partially adapted from the [sklearn documentation](https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html)

# In[ ]:


import pandas as pd
import numpy as np

from scipy.io import arff
from scipy.stats import randint, uniform

import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
# from sklearn.metrics import log_loss
from sklearn.metrics import roc_curve, auc, roc_auc_score

from time import time
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

rnd_seed = 11
n_folds = 3


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


# Make Train, Validation, and Test Sets

# In[ ]:


X = df[features].values
y = df[target].values

X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, random_state=rnd_seed, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.2, random_state=rnd_seed+1, stratify=y_tmp)
del X_tmp; del y_tmp


# Prepare Stratified k-Folds

# In[ ]:


skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=rnd_seed+2)


# ## Set Hyperparameter Rangess
# #### See the [docs here](https://xgboost.readthedocs.io/en/latest/parameter.html) for XGBoost hyperparameter details.

# In[ ]:


max_num_boost_rounds = 200 # maximum number of boosting rounds to run / trees to create
num_early_stopping_rounds = 10 # must see improvement over last num_early_stopping_rounds or will halt
xgb_objective = 'binary:logistic'
xgb_verbosity = 0 #  The degree of verbosity. Valid values are 0 (silent) - 3 (debug).
xgb_n_jobs = 1 # Number of parallel threads used to run XGBoost. -1 makes use of all cores in your system
# search_scoring = 'roc_auc' # need to use custom function to work properly with xgb early stopping, see xgb_early_stopping_auc_scorer
search_n_jobs = 2 # Number of parallel threads used to run hyperparameter searches
search_verbosity = 1


# In[ ]:


fixed_fit_params = {
    'early_stopping_rounds': num_early_stopping_rounds,
    'eval_set': [(X_val, y_val)],
    'eval_metric': 'logloss',
}


# In[ ]:


param_dists = {
    'max_depth': randint(3, 10), # default=6, Maximum depth of a tree. Increasing this value will make the model more complex and more likely to overfit.
    'min_child_weight': uniform(1.0, 5.0), # default=1, Minimum sum of instance weight (hessian) needed in a child. If the tree partition step results in a leaf node with the sum of instance weight less than min_child_weight, then the building process will give up further partitioning. In linear regression task, this simply corresponds to minimum number of instances needed to be in each node. The larger min_child_weight is, the more conservative the algorithm will be.
    'learning_rate': uniform(0.05, 1.0), # default=0.3, Step size shrinkage used in update to prevents overfitting. After each boosting step, we can directly get the weights of new features, and eta shrinks the feature weights to make the boosting process more conservative. alias: learning_rate
    # 'gamma': uniform(1, 5), # default=0, Minimum loss reduction required to make a further partition on a leaf node of the tree. The larger gamma is, the more conservative the algorithm will be. alias: min_split_loss
    # 'colsample_bytree': uniform(0.5, 1.0), # default=1, Subsample ratio of columns when constructing each tree.
    # 'subsample': uniform(0.5, 1.0), # default=1, Subsample ratio of the training instances. Setting it to 0.5 means that XGBoost would randomly sample half of the training data prior to growing trees. and this will prevent overfitting. Subsampling will occur once in every boosting iteration.
    # 'max_delta_step': uniform(0.0, 5.0), # default=0, Maximum delta step we allow each leaf output to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative. Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced. Set it to value of 1-10 might help control the update.
    # 'reg_alpha': uniform(0.0, 5.0), # default=0, L1 regularization term on weights. Increasing this value will make model more conservative.
    # 'reg_lambda': uniform(0.0, 5.0), # default=1, L2 regularization term on weights. Increasing this value will make model more conservative.
}


# # Random Search

# In[ ]:


search_n_iter=50
model_rs = xgb.XGBClassifier(n_estimators=max_num_boost_rounds, objective=xgb_objective, verbosity=xgb_verbosity)

rs = RandomizedSearchCV(estimator=model_rs, param_distributions=param_dists, scoring=xgb_early_stopping_auc_scorer,
                        n_iter=search_n_iter, n_jobs=search_n_jobs, cv=skf, verbose=search_verbosity, random_state=rnd_seed+3
                       )


# In[ ]:


rs_start = time()

rs.fit(X_train, y_train, groups=None, **fixed_fit_params)

rs_time = time()-rs_start

print(f'RandomizedSearchCV took {rs_time:.2f} seconds for {search_n_iter} candidates parameter settings')


# In[ ]:


report(rs)


# In[ ]:


# search time, best model, best params, for later comparison
# rs_time
# rs.best_estimator_
# rs.best_params_


# # Grid Search TODO

# # Bayesian Optimization TODO

# # Dev

# In[ ]:


# stand alone
model = xgb.XGBClassifier(max_depth=6, verbosity=0)
model.fit(X_train, y_train, early_stopping_rounds=num_early_stopping_rounds, eval_set=[(X_val, y_val)], eval_metric='logloss')
y_test_pred = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:,1]


# In[ ]:


# best from rs
model = rs.best_estimator_
y_test_pred = model.predict_proba(X_test, ntree_limit=model.best_ntree_limit)[:,1]


# # Evaluate Performance TODO

# In[ ]:


def plot_y_pred(y_pred, y):
    fig, ax = plt.subplots()

    sig_mask = np.where(y == 1)
    bkg_mask = np.where(y == 0)

    plt.hist(y_pred[sig_mask], bins=np.linspace(0,1,11), histtype='step', color='C0', label='Signal')
    plt.hist(y_pred[bkg_mask], bins=np.linspace(0,1,11), histtype='step', color='C1', label='Background')

    leg = ax.legend(loc='upper right',frameon=False)
    leg.get_frame().set_facecolor('none')
    ax.set_yscale('log')

    ax.set_xlabel('$\hat{y}$')
    ax.set_ylabel('Counts')
    ax.set_xlim([0.,1.])

    plt.show()
    # fig.savefig('roc.pdf')


# In[ ]:


plot_y_pred(y_test_pred, y_test)


# In[ ]:


fpr, tpr, thr = roc_curve(y_test, y_test_pred)


# In[ ]:


def plot_roc(fpr, tpr, rndGuess=True, grid=False, better_ann=True):
    fig, ax = plt.subplots()

    label=f'AUC {auc(fpr,tpr):.4f}'

    ax.plot(fpr, tpr, lw=2, c='C0', ls='-', label=label)

    if rndGuess:
        x = np.linspace(0., 1.)
        ax.plot(x, x, color='grey', linestyle=':', linewidth=2, label='Random Guess')

    if grid:
        ax.grid()

    leg = ax.legend(loc='lower right',frameon=False)
    leg.get_frame().set_facecolor('none')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlim([0.,1.])
    ax.set_ylim([0.,1.])

    if better_ann:
        plt.text(-0.08, 1.08, 'Better', size=12, rotation=45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

    plt.show()
    # fig.savefig('roc.pdf')


# In[ ]:


plot_roc(fpr, tpr)

