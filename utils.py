import os
import pickle
import pandas as pd
import numpy as np

########################################################
# dump results to pkl
def dump_to_pkl(results, fname, m_path='pkl_results', tag=''):
	os.makedirs(m_path, exist_ok=True)
	with open(f'{m_path}/{fname}{tag}.pkl', 'wb') as f:
		pickle.dump( results, f )

########################################################
# dump results to pkl
def load_from_pkl(fname, m_path='pkl_results', tag=''):
	results = None
	with open(f'{m_path}/{fname}{tag}.pkl', 'rb') as f:
		results = pickle.load( f )
	return results

########################################################
# Report best scores from sklearn searches
def report(results, n_top=3):
	results = results.cv_results_
	for i in range(1, n_top+1):
		candidates = np.flatnonzero(results['rank_test_score'] == i)
		for candidate in candidates:
			print(f'Model with rank: {i}')
			print(f"Mean validation score: {results['mean_test_score'][candidate]:.3f} (std: {results['std_test_score'][candidate]:.3f})")
			print('Parameters: {0}\n'.format(results['params'][candidate]))

########################################################
# helper function for _to_csv functions
def _df_to_csv(df, params_to_be_opt, m_path, tag):
	os.makedirs(m_path, exist_ok=True)

	df = df.reset_index().rename(columns={'index': 'iter'})
	df = df[['iter', 'y', 'auc']+params_to_be_opt]
	df.to_csv(f'{m_path}/iter_results{tag}.csv', index=False, na_rep='nan')

	df_best = df.copy()
	df_best = df_best.loc[df_best['y'].min() == df_best['y']]
	df_best = df_best.drop_duplicates(subset=params_to_be_opt).reset_index(drop=True)
	df_best = df_best.sort_values(by=params_to_be_opt).reset_index(drop=True)
	df_best = df_best[['y', 'auc']+params_to_be_opt]
	df_best.to_csv(f'{m_path}/best_params_points{tag}.csv', index=False, na_rep='nan')

########################################################
# Save iteration results from sklearn searches
def output_sklearn_to_csv(sklearn_result, params_to_be_opt, m_path='output', tag=''):
	cols_dict = {}
	cols_dict['y'] = [-y for y in sklearn_result.cv_results_['mean_test_score']]
	for param in params_to_be_opt:
		cols_dict[param] = list(sklearn_result.cv_results_[f'param_{param}'])

	df = pd.DataFrame(cols_dict)
	df['auc'] = -df['y']

	_df_to_csv(df, params_to_be_opt, m_path, tag)

########################################################
# Save iteration results from hyperopt searches
def output_hyperopt_to_csv(hyperopt_result, params_to_be_opt, m_path='output', tag=''):
	cols_dict = {}
	cols_dict['y'] = hyperopt_result.losses()

	for param in params_to_be_opt:
		cols_dict[param] = hyperopt_result.vals[param]

	df = pd.DataFrame(cols_dict)
	df['auc'] = -df['y']

	_df_to_csv(df, params_to_be_opt, m_path, tag)

########################################################
# Save iteration results from gentun searches
def output_gentun_to_csv(_df, params_to_be_opt, m_path='output', tag=''):
	df = _df.copy()
	df = df.rename(columns={'best_fitness': 'auc'})
	df['y'] = -df['auc']

	_df_to_csv(df, params_to_be_opt, m_path, tag)

########################################################
def combine_best_results(optimizer_abbrevs, params_to_be_opt, params_initial, y_initial, m_path='output', fname='all_best_results'):
	def _load_df(_fname, tag='', m_path=m_path, cols_int=[], cols_str=[]):
		full_fname = f'{m_path}/{_fname}{tag}.csv'
		try:
			df = pd.read_csv(full_fname)
			for col in df.columns:
				if col in cols_int:
					df[col] = df[col].astype(int)
				elif col in cols_str:
					df[col] = df[col].astype(str)
				else:
					df[col] = df[col].astype(float)
			return df
		except:
			raise ValueError('Could not open csv!')

	df_best_results = None

	for opt_name in optimizer_abbrevs:
		df = _load_df('best_params_points', tag=f'_{opt_name}', cols_int=['max_depth'])
		df['optimizer'] = opt_name
		if df_best_results is None:
			df_best_results = df.copy()
		else:
			df_best_results = pd.concat([df_best_results, df])

	initial_row = {'optimizer': 'Initial', 'y': y_initial, 'auc': -y_initial}
	for param in params_to_be_opt:
		initial_row[param] = params_initial[param]
	df_best_results = df_best_results.append(initial_row, ignore_index=True)

	df_best_results['per_change'] = 100.*(y_initial - df_best_results['y'])/y_initial

	df_best_results = df_best_results.sort_values(by='y', ascending=True).reset_index(drop=True)
	fixed_cols = ['optimizer', 'y', 'per_change', 'auc']+params_to_be_opt
	cols = fixed_cols + list(set(df_best_results.columns)-set(fixed_cols))
	df_best_results = df_best_results[cols]

	df_best_results.to_csv(f'{m_path}/{fname}.csv', index=False, na_rep='nan')

	return df_best_results
