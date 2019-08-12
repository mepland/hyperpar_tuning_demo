# python
import os
import pandas as pd
import numpy as np

from sklearn.metrics import auc

########################################################
# plotting
import matplotlib as mpl
mpl.use('Agg', warn=False)
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
# import matplotlib.cm as cm
# from matplotlib import gridspec
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import matplotlib.ticker as ticker

########################################################
# Set common plot parameters
vsize = 11 # inches
# aspect ratio width / height
aspect_ratio_single = 4.0/3.0
aspect_ratio_multi = 1.0

plot_png=True
png_dpi=500

std_ann_x = 0.80
std_ann_y = 0.94

# std_cmap = cm.plasma
# std_cmap_r = cm.plasma_r

########################################################
def plot_bo_opt_convergence(y_values, y_title, x_bests, ann_text, m_path, fname, do_min=True, y_initial=None, tag='', inline=False):

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
	df_values_best = df_values.loc[df_values['x'].isin(x_bests)]
	df_values_reg = df_values.loc[np.logical_not(df_values['x'].isin(x_bests))]

	if do_min:
		y_best = df_values['y'].min()
	else:
		y_best = df_values['y'].max()

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
			ann_text_final = f'{ann_text}\n{y_title} Inital: {y_initial:.5f}\nBest: {y_best:.5f}\nDecrease: {y_initial-y_best:.5f}\nPercentage: {(y_initial-y_best)/y_initial:.2%}'
		else:
			ann_text_final = f'{ann_text}\n{y_title} Inital: {y_initial:.5f}\nBest: {y_best:.5f}\nIncrease: {y_best-y_initial:.5f}\nPercentage: {(y_best-y_initial)/y_initial:.2%}'
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
def plot_y_pred(y_pred, y, m_path, fname='y_pred', tag='', ann_text=None, inline=False):
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
		plt.figtext(std_ann_x, std_ann_y, ann_text, ha='center', va='top', size=18, zorder=3)

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
# TODO revive overlapping one
def plot_roc(fpr, tpr, m_path, fname='y_pred', tag='', rndGuess=True, grid=False, better_ann=True, ann_text=None, inline=False):
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

	plt.tight_layout()
	if inline:
		fig.show()
	else:
		os.makedirs(m_path, exist_ok=True)
		if plot_png:
			fig.savefig(f'{m_path}/{fname}{tag}.png', dpi=png_dpi)
		fig.savefig(f'{m_path}/{fname}{tag}.pdf')
		plt.close('all')
