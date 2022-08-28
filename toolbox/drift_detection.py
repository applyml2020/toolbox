# drift detection
from .preprocessing import Preprocessing
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

# define class for automating distribution plot analysis
class DriftDetection(Preprocessing):
	# initialize
	def __init__(self, cls_model_preprocessing, str_dirname_output='/opt/ml/model', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		Preprocessing.__init__(self, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.cls_model_preprocessing = cls_model_preprocessing
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# preprocess data
	def preprocess_data(self, X):
		# transform
		X = self.cls_model_preprocessing.transform(X)
		# return X
		return X
	# detect drift
	def median_distributions(self, list_cols, df_train, df_valid, df_test):
		# create string for new dir
		str_new_dir = f'{self.str_dirname_output}/median_distribution_plots'
		# create dir for output
		try:
			os.mkdir(str_new_dir)
		except FileExistsError:
			pass
		# iterate through cols
		list_sig_diff = []
		for col in tqdm (list_cols):
			# create a df with just the col
			df_col = pd.DataFrame({
				'train': list(df_train[col]),
				'valid': list(df_valid[col]),
				'test': list(df_test[col]),
			})
			# get nrows
			int_n_rows = df_col.shape[0]
			# get n samples
			int_n_samples = 100 # hard code for now
			# get number of rows per sample
			int_len_sample = int(int_n_rows / int_n_samples)
			# create list to use for sample
			list_empty = []
			for a in range(int_n_samples):
				# create list containing value for b the same length as a samplel
				list_ = list(itertools.repeat(a, int_len_sample))
				# extend list_empty
				list_empty.extend(list_)
			# create a dictionary to use for grouping
			dict_agg = {col: 'median' for col in df_col.columns}
			# make list_empty into a column in df_col
			df_col['sample'] = list_empty
			# group df_col by sample and get median for each of 100 samples
			df_col = df_col.groupby('sample', as_index=False).agg(dict_agg)

			# make distribution plot
			fig, ax = plt.subplots()
			ax.set_title(f'{col}')
			sns.histplot(df_col['train'], kde=False, color="r", ax=ax, label='Train')
			sns.histplot(df_col['valid'], kde=False, color="g", ax=ax, label='Valid')
			sns.histplot(df_col['test'], kde=False, color="b", ax=ax, label='Test')
			plt.legend()
			plt.savefig(f'{str_new_dir}/{col}.png', bbox_inches='tight')
			plt.close()
		# return object
		return self