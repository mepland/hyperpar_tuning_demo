# python
import os
import pandas as pd
import numpy as np
import math
from collections import OrderedDict

from sklearn.metrics import auc
from skopt.plots import _format_scatter_plot_axes, partial_dependence

########################################################
# class conversions
import skopt
from scipy.optimize import OptimizeResult
from skopt.utils import create_result
import hyperopt
import neptunecontrib.hpo.utils as hpo_utils
import sklearn

def _convert_to_skopt_OptimizeResult(_result_in):
	def _my_create_result(bo_opt):
		return create_result(bo_opt.Xi, bo_opt.yi, bo_opt.space, bo_opt.rng, models=bo_opt.models)

	if isinstance(_result_in, skopt.optimizer.optimizer.Optimizer):
		# _result_in = skopt.optimizer.optimizer.Optimizer, from skopt

		return _my_create_result(_result_in)

	elif isinstance(_result_in, tuple):
		optimize_results = OptimizeResult()

		if isinstance(_result_in[0], hyperopt.base.Trials):
			# _results_in = (hyperopt.base.Trials, collections.OrderedDict), from hyperopt

			# hyperopt2skopt doesn't quite make the right OptimizeResult, but does handle spaces decently, so used it
			# see https://neptune-contrib.readthedocs.io/user_guide/hpo/utils.html

			bo_opt = hpo_utils.hyperopt2skopt(_result_in[0], _result_in[1])

			optimize_results.Xi = bo_opt.x_iters
			optimize_results.yi = bo_opt.func_vals
			optimize_results.space = bo_opt.space


		elif isinstance(_result_in[0], sklearn.model_selection._search.RandomizedSearchCV) or isinstance(_result_in[0], sklearn.model_selection._search.GridSearchCV):
			# _results_in = (RandomizedSearchCV or GridSearchCV, collections.OrderedDict), results from sklearn, with a matching space from hyperopt - bit of a hack...

			Xi = []
			for params in _result_in[0].cv_results_['params']:
				Xi.append([params[param] for param in list(_result_in[1].keys())])

			optimize_results.Xi = Xi
			optimize_results.yi = [-score for score in _result_in[0].cv_results_['mean_test_score']]
			optimize_results.space = hpo_utils._convert_space_hop_skopt(_result_in[1])

		else:
			raise ValueError(f"Don't know how to handle {type(_result_in)}!!")

		optimize_results.models = None # needed for my_plot_objective
		optimize_results.rng = None # not needed
		return _my_create_result(optimize_results)

	else:
		raise ValueError(f"Don't know how to handle {type(_result_in)}!!")

########################################################
# plotting
import matplotlib as mpl
# mpl.use('Agg', warn=False)
# mpl.rcParams['font.family'] = ['HelveticaNeue-Light', 'Helvetica Neue Light', 'Helvetica Neue', 'Helvetica', 'Arial', 'Lucida Grande', 'sans-serif']
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.top']           = True
mpl.rcParams['ytick.right']         = True
mpl.rcParams['xtick.direction']     = 'in'
mpl.rcParams['ytick.direction']     = 'in'
mpl.rcParams['xtick.labelsize']     = 13
mpl.rcParams['ytick.labelsize']     = 13
mpl.rcParams['xtick.minor.visible'] = True
mpl.rcParams['ytick.minor.visible'] = True
mpl.rcParams['xtick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['xtick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['xtick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['xtick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['xtick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['xtick.minor.pad']     = 1.4  # distance to the minor tick label in points
mpl.rcParams['ytick.major.width']   = 0.8  # major tick width in points
mpl.rcParams['ytick.minor.width']   = 0.8  # minor tick width in points
mpl.rcParams['ytick.major.size']    = 7.0  # major tick size in points
mpl.rcParams['ytick.minor.size']    = 4.0  # minor tick size in points
mpl.rcParams['ytick.major.pad']     = 1.5  # distance to major tick label in points
mpl.rcParams['ytick.minor.pad']     = 1.4  # distance to the minor tick label in points
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from matplotlib import gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.ticker as ticker
from matplotlib.ticker import LogLocator

########################################################
# Set common plot parameters
vsize = 11 # inches
# aspect ratio width / height
aspect_ratio_single = 4./3.
aspect_ratio_multi = 1.

plot_png=True
png_dpi=500

std_ann_x = 0.80
std_ann_y = 0.94

# std_cmap = cm.plasma
# std_cmap_r = cm.plasma_r

########################################################
def plot_convergence(y_values, ann_text, y_title='$y$', m_path='output', fname='convergence_y', tag='', do_min=True, y_initial=None, inline=False):

	def cumulative_best(xs):
		result = np.zeros_like(xs)
		cbest = xs[0]
		result[0] = xs[0]
		for i in range(1, xs.shape[0]):
			if (do_min and cbest > xs[i]) or (not do_min and cbest < xs[i]):
				cbest = xs[i]
			result[i] = cbest
		return result

	df_values = pd.DataFrame({'x':np.arange(len(y_values)), 'y':y_values})

	if do_min:
		y_best = df_values['y'].min()
	else:
		y_best = df_values['y'].max()

	x_bests = np.where(y_best == np.array(y_values))

	df_values_best = df_values.loc[df_values['x'].isin(x_bests)]
	df_values_reg = df_values.loc[np.logical_not(df_values['x'].isin(x_bests))]

	fig, ax = plt.subplots()
	fig.set_size_inches(aspect_ratio_single*vsize, vsize)
	ax.set_xlabel('Iteration')
	ax.set_ylabel(y_title)
	if y_initial is not None:
		init = ax.axhline(y=y_initial, label='Initial', color='black', alpha=0.3, linestyle='--', linewidth=2, zorder=0)

	ax.plot(cumulative_best(df_values['y']), label='Best Discovered', color='black', linewidth=2, zorder=0)

	ax.scatter(df_values_reg['x'], df_values_reg['y'], label='Observations', s=140, facecolors='blue', zorder=1)
	ax.scatter(df_values_best['x'], df_values_best['y'], label='Best', s=140, facecolors='green', zorder=2)

	xmin,xmax = ax.get_xlim()
	ax.set_xlim(xmin, xmin+1.3*(xmax-xmin))
	ymin,ymax = ax.get_ylim()
	ax.set_ylim(ymin, ymin+1.1*(ymax-ymin))

	leg = ax.legend(loc='upper right', fontsize = 18)
	leg.get_frame().set_edgecolor('none')
	leg.get_frame().set_facecolor('white')

	if y_initial is not None:
		if do_min:
			ann_text_final = f'{ann_text}\n{y_title} Initial: {y_initial:.5f}\nBest: {y_best:.5f}\nDecrease: {y_initial-y_best:.5f}\nPercentage: {(y_initial-y_best)/y_initial:.2%}'
		else:
			ann_text_final = f'{ann_text}\n{y_title} Initial: {y_initial:.5f}\nBest: {y_best:.5f}\nIncrease: {y_best-y_initial:.5f}\nPercentage: {(y_best-y_initial)/y_initial:.2%}'
	else:
		ann_text_final = f'{ann_text}\n{y_title} Best: {y_best:.5f}'

	plt.figtext(0.83, 0.82, ann_text_final, ha='center', va='top', size=18, zorder=3)

	plt.tight_layout()
	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')

########################################################
def plot_y_pred(y_pred, y, m_path='output', fname='y_pred', tag='', ann_text=None, inline=False):
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

	if ann_text is not None:
		plt.figtext(std_ann_x, std_ann_y, ann_text, ha='center', va='top', size=18)

	plt.tight_layout()
	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')

########################################################
# plot overlaid roc curves for many models
def plot_rocs(models, m_path='output', fname='roc', tag='', rndGuess=False, better_ann=True, grid=False, inverse_log=False, x_axis_params=None, y_axis_params=None, inline=False):
	fig, ax = plt.subplots()

	if rndGuess:
		if inverse_log:
			x = np.linspace(1e-10, 1.)
			ax.plot(x, 1/x, color='grey', linestyle=':', linewidth=2, label='Random Guess')
		else:
			x = np.linspace(0., 1.)
			ax.plot(x, x, color='grey', linestyle=':', linewidth=2, label='Random Guess')

	for model in models:
		# models is a list of dicts with keys name, nname, fpr, tpr, c (color), ls (linestyle)

		if inverse_log:
			with np.errstate(divide='ignore'):
				y_values = np.divide(1., model['tpr'])
		else:
			y_values = model['tpr']

		label=f"{model['nname']}, AUC: {auc(model['fpr'],model['tpr']):.4f}"

		ax.plot(model['fpr'], y_values, lw=2, c=model.get('c', 'blue'), ls=model.get('ls', '-'), label=label)

		fname = f"{fname}_{model['name']}"

	if grid:
		ax.grid()

	if inverse_log:
		leg_loc = 'upper right'
	else:
		leg_loc = 'lower right'

	leg = ax.legend(loc=leg_loc,frameon=False)
	leg.get_frame().set_facecolor('none')

	ax.set_xlim([0.,1.])
	ax.set_xlabel('False Positive Rate')

	# TODo see if still needed
	# x_labels = ['{:}'.format(float(x)) for x in ax.get_xticks().tolist()]
	# x_labels[0] = ''
	# ax.set_xticklabels(x_labels)

	if inverse_log:
		ax.set_yscale('log')
		ax.set_ylabel('Inverse True Positive Rate')
	else:
		ax.set_xlim([0.,1.])
		ax.set_ylabel('True Positive Rate')

	if not isinstance(x_axis_params, dict):
		x_axis_params = dict()
	x_min_current, x_max_current = ax.get_xlim()
	x_min = x_axis_params.get('min', x_min_current)
	x_max = x_axis_params.get('max', x_max_current)
	ax.set_xlim([x_min, x_max])

	if not isinstance(y_axis_params, dict):
		y_axis_params = dict()
	y_min_current, y_max_current = ax.get_ylim()
	y_min = y_axis_params.get('min', y_min_current)
	y_max = y_axis_params.get('max', y_max_current)
	ax.set_ylim([y_min, y_max])

	if better_ann:
		if inverse_log:
			plt.text(-0.07, -0.12, 'Better', size=12, rotation=-45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))
		else:
			plt.text(-0.07, 1.08, 'Better', size=12, rotation=45, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='green', alpha=0.2))

	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')

########################################################
########################################################
# The functions below are modified versions of the standard skopt functions in:
# https://github.com/scikit-optimize/scikit-optimize/blob/master/skopt/plots.py
########################################################
########################################################

########################################################
def my_plot_evaluations(_result_in, m_path='output', fname='evaluation', tag='', bins=20, dimensions=None, ann_text=None, inline=False):
	result = _convert_to_skopt_OptimizeResult(_result_in)

	space = result.space
	samples = np.asarray(result.x_iters)
	order = range(samples.shape[0])
	fig, ax = plt.subplots(space.n_dims, space.n_dims, figsize=(2 * space.n_dims, 2 * space.n_dims))
	fig.set_size_inches(aspect_ratio_single*vsize, vsize)
	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)

	leg_objects = []

	for i in range(space.n_dims):
		for j in range(space.n_dims):
			if i == j:
				if space.dimensions[j].prior == 'log-uniform':
					low, high = space.bounds[j]
					bins_ = np.logspace(np.log10(low), np.log10(high), bins)
				else:
					bins_ = bins
				ax[i, i].hist(samples[:, j], bins=bins_, range=space.dimensions[j].bounds)

			# lower triangle
			elif i > j:
				ax[i, j].scatter(samples[:, j], samples[:, i], c=order, s=40, lw=0., cmap=cm.viridis)
				ax[i, j].scatter(result.x[j], result.x[i], c=['r'], s=20, lw=0.)

	_ = _format_scatter_plot_axes(ax, space, ylabel='$N$', dim_labels=dimensions)

	norm = mpl.colors.Normalize(vmin=0., vmax=int(math.ceil(max(order))/10.)*10.)
	cax = fig.add_axes([0.48, 0.82, 0.42, 0.05], label='cax')
	cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm.viridis, norm=norm, orientation='horizontal', label='Iteration')

	leg_objects.append(plt.Line2D([0],[0], ls='None', marker='o', c='r', ms=12, label='Best'))

	if len(leg_objects) > 0:
		leg = fig.legend(leg_objects, [ob.get_label() for ob in leg_objects], fontsize=18, bbox_to_anchor=(0.76, 0.55, 0.2, 0.2), loc='upper left', ncol=1, borderaxespad=0.)
		leg.get_frame().set_edgecolor('none')
		leg.get_frame().set_facecolor('none')

	if ann_text is not None:
		plt.figtext(std_ann_x, std_ann_y, ann_text, ha='center', va='top', size=18)

	# increase margins
	fig.subplots_adjust(
		left = 0.125,  # the left side of the subplots of the figure
		right = 0.9,   # the right side of the subplots of the figure
		bottom = 0.1,  # the bottom of the subplots of the figure
		top = 0.9,     # the top of the subplots of the figure
		wspace = 0.2,  # the amount of width reserved for space between subplots, expressed as a fraction of the average axis width
		hspace = 0.2,  # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height)
	)

	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')

########################################################
def my_plot_objective(_result_in, m_path='output', fname='objective', tag='', levels=10, n_points=40, n_samples=250, size=2, zscale='linear', dimensions=None, ann_text=None, inline=False):
	result = _convert_to_skopt_OptimizeResult(_result_in)

	if result.models is None:
		raise ValueError("result.models is None, can't plot_objective. _results_in is probably from hyperopt and is not supported")

	space = result.space
	samples = np.asarray(result.x_iters)
	rvs_transformed = space.transform(space.rvs(n_samples=n_samples))

	if zscale == 'log':
		locator = LogLocator()
	elif zscale == 'linear':
		locator = None
	else:
		raise ValueError(f'Valid values for zscale are linear and log, not {zscale}')

	fig, ax = plt.subplots(space.n_dims, space.n_dims, figsize=(size * space.n_dims, size * space.n_dims))
	fig.set_size_inches(aspect_ratio_single*vsize, vsize)
	fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95, hspace=0.1, wspace=0.1)

	leg_objects = []
	zi_min = None
	zi_max = None

	for i in range(space.n_dims):
		for j in range(space.n_dims):
			if i == j:
				xi, yi = partial_dependence(space, result.models[-1], i, j=None, sample_points=rvs_transformed, n_points=n_points)
				ax[i, i].plot(xi, yi)
				ax[i, i].axvline(result.x[i], linestyle='--', color='r', lw=1)
			# lower triangle
			elif i > j:
				xi, yi, zi = partial_dependence(space, result.models[-1], i, j, rvs_transformed, n_points)
				if zi_min is None:
					zi_min = np.min(zi)
				else:
					zi_min = min(zi_min, np.min(zi))
				if zi_max is None:
					zi_max = np.max(zi)
				else:
					zi_max = max(zi_max, np.max(zi))
				ax[i, j].contourf(xi, yi, zi, levels, locator=locator, cmap=cm.viridis_r)
				ax[i, j].scatter(samples[:, j], samples[:, i], c='k', s=10, lw=0.)
				ax[i, j].scatter(result.x[j], result.x[i], c=['r'], s=20, lw=0.)

	_ = _format_scatter_plot_axes(ax, space, ylabel='$\partial($dependence$)$', dim_labels=dimensions)

	norm = mpl.colors.Normalize(vmin=zi_min, vmax=zi_max)
	cax = fig.add_axes([0.48, 0.82, 0.42, 0.05], label='cax')
	cb1 = mpl.colorbar.ColorbarBase(cax, cmap=cm.viridis, norm=norm, orientation='horizontal', label='$y$')

	leg_objects.append(plt.Line2D([0],[0], ls='None', marker='o', c='black', ms=12, label='Iterations'))
	leg_objects.append(plt.Line2D([0],[0], ls='None', marker='o', c='r', ms=12, label='Best'))

	if ann_text is not None:
		plt.figtext(std_ann_x, std_ann_y, ann_text, ha='center', va='top', size=18)

	if len(leg_objects) > 0:
		leg = fig.legend(leg_objects, [ob.get_label() for ob in leg_objects], fontsize=18, bbox_to_anchor=(0.76, 0.55, 0.2, 0.2), loc='upper left', ncol=1, borderaxespad=0.)
		leg.get_frame().set_edgecolor('none')
		leg.get_frame().set_facecolor('none')

	# increase margins
	fig.subplots_adjust(
		left = 0.125,  # the left side of the subplots of the figure
		right = 0.9,   # the right side of the subplots of the figure
		bottom = 0.1,  # the bottom of the subplots of the figure
		top = 0.9,     # the top of the subplots of the figure
		wspace = 0.2,  # the amount of width reserved for space between subplots, expressed as a fraction of the average axis width
		hspace = 0.2,  # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height)
	)

	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')
