# train valid test split
from .exploratory_data_analysis import ExploratoryDataAnalysis
import numpy as np
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm

# define class
class TrainValidTestSplit(ExploratoryDataAnalysis):
	# initialize
	def __init__(self, str_dirname_output='./output', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		ExploratoryDataAnalysis.__init__(self, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# train, valid, test split
	def train_valid_test_split(self, df, flt_prop_train=0.50, flt_prop_val_test=0.25, str_filepath='s3://gen-xii/data/gen-xii/pd'):
		# number of rows in df
		int_nrows = df.shape[0]
		# list for sections
		list_sections = [
		    int(flt_prop_train * int_nrows),
		    int((flt_prop_train + flt_prop_val_test) * int_nrows),
		]
		# split
		df_train, df_valid, df_test = np.split(
		    ary=df,
		    indices_or_sections=list_sections,
		    axis=0, # split on rows, not columns
		)
		# get non-numeri cols
		list_nonnumeric = [col for col in df_train.columns if not is_numeric_dtype(df_train[col])]
		# make sure date col is excluded
		list_nonnumeric = [col for col in list_nonnumeric if col != self.str_datecol]
		# save to parquet
		df_train[list_nonnumeric] = df_train[list_nonnumeric].astype(str)
		df_train.to_parquet(f'{str_filepath}/df_train.gzip', compression='gzip')
		# save to parquet
		df_valid[list_nonnumeric] = df_valid[list_nonnumeric].astype(str)
		df_valid.to_parquet(f'{str_filepath}/df_valid.gzip', compression='gzip')
		# save to parquet
		df_test[list_nonnumeric] = df_test[list_nonnumeric].astype(str)
		df_test.to_parquet(f'{str_filepath}/df_test.gzip', compression='gzip')
		# return object
		return self
	# identify all nan
	def find_all_nan_columns(self, df, list_cols):
		# iterate
		list_cols_all_nan = []
		for col in tqdm (list_cols):
			# get total nan
			int_sum_nan = df[col].isnull().sum()
			# logic
			if int_sum_nan == len(df[col]):
				# append
				list_cols_all_nan.append(col)
		# get those where all rows are NaN
		self.list_cols_all_nan = list_cols_all_nan
		# return object
		return self
	# drop all nan
	def drop_all_nan_columns(self, df):
		# drop
		df = df.drop(self.list_cols_all_nan, axis=1, inplace=False)
		# return df
		return df
	# find no variance columns
	def find_no_variance_columns(self, df, list_cols):
		# iterate
		list_cols_no_variance = []
		for col in tqdm (list_cols):
			# get nunique
			int_n_unique = df[col].nunique()
			# logic
			if int_n_unique == 1:
				# append to list
				list_cols_no_variance.append(col)
		# get nunique 1
		self.list_cols_no_variance = list_cols_no_variance
		# return object
		return self
	# drop no variance columns
	def drop_no_variance_columns(self, df):
		# drop
		df = df.drop(self.list_cols_no_variance, axis=1, inplace=False)
		# return df
		return df
	# find redundant features
	def find_redundant_columns(self, df, list_cols, int_n_rows_check=1000):
		# instantiate empty list
		list_cols_redundant = []
		a = 0
		for cola in tqdm (list_cols):
			# logic
			if cola not in list_cols_redundant:
				# iterate through the other cols
				for colb in list_cols[a+1:]:
					# check if subset of cola == colb
					if df[cola].iloc[:int_n_rows_check].equals(df[colb].iloc[:int_n_rows_check]):
						# check if the whole column is redundant
						if df[cola].equals(df[colb]):
							# print message
							#print(f'After checking all rows, {colb} is redundant with {cola}')
							list_cols_redundant.append(colb)
			# increase a
			a += 1
		# save to object
		self.list_cols_redundant = list_cols_redundant
		# return object
		return self
	# drop redundant columns
	def drop_redundant_columns(self, df):
		# drop
		df = df.drop(self.list_cols_redundant, axis=1, inplace=False)
		# return df
		return df
