# preprocessing
from .create_constants import CreateConstants
from sklearn.base import BaseEstimator, TransformerMixin
import time
import pandas as pd
import numpy as np
from tqdm import tqdm

# define class
class Preprocessing(CreateConstants):
	# initialize
	def __init__(self, str_dirname_output='./output', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		CreateConstants.__init__(self, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# show transformers
	def show_transformers(self, list_transformers):
		# iterate
		for a, transformer in enumerate(list_transformers):
			print(f'{a+1}: {transformer.__class__.__name__} - {transformer.str_message}')
		# return object
		return self

# data type setter
class SetDataTypes(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, bool_obj_to_string=True, bool_iterate=True, bool_verbose=True, str_message='Data Type Setter'):
		self.bool_obj_to_string = bool_obj_to_string
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		# get the data types into a dictionary
		dict_dtypes = dict(X.dtypes)
		# save to object
		self.dict_dtypes = dict_dtypes
		return self
	# transform
	def transform(self, X):
		# fillna
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_dtypes = {key: val for key, val in self.dict_dtypes.items() if key in list(X.columns)}
		# if setting to string
		if self.bool_obj_to_string:
			# change O to str
			dict_dtypes = {key: ('str' if val == 'O' else val) for key, val in dict_dtypes.items()}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_dtypes.items()):
				X[key] = X[key].astype(val)
		# if not iterating
		else:
			X = X.astype(dict_dtypes)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# class for inflation
class Inflator(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, dict_inflation_rate, dict_replace_errors, bool_iterate=True, bool_verbose=True, str_datecol='applicationdate__app', str_message='Inflator', bool_rm_error_codes=True, bool_drop=True):
		self.list_cols = list_cols
		self.dict_inflation_rate = dict_inflation_rate
		self.dict_replace_errors = dict_replace_errors
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_datecol = str_datecol
		self.str_message = str_message
		self.bool_rm_error_codes = bool_rm_error_codes
		self.bool_drop = bool_drop
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]

		# create year
		X['year'] = X[self.str_datecol].dt.year
		# map factor to year
		X['factor'] = X['year'].map(self.dict_inflation_rate)

		# convert
		if self.bool_iterate:
			for col in tqdm (list_cols):
				# multiply by factor
				X[col] = X[col] * X['factor']
		else:
			X[list_cols] = X[list_cols].multiply(X['factor'], axis='index')

		# replace error codes
		if self.bool_rm_error_codes:
			# replace
			X[list_cols] = X[list_cols].replace(self.dict_replace_errors, inplace=False)
		else:
			pass

		# drop 
		if self.bool_drop:
			X = X.drop(['year','factor'], axis=1, inplace=False)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return
		return X

# class for cleaning text
class CleanText(BaseEstimator, TransformerMixin):
	# initialize class
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True, str_message='Text Cleaner'):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# define helper function
		def clean_text(str_text):
			# if NaN
			if not pd.isnull(str_text):
				# strip leading/trailing whitespace, rm spaces, lower
				return str(str_text).strip().replace(' ', '').lower()
			else:
				return str_text

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# if iterating
		if self.bool_iterate:
			for col in tqdm (list_cols):
				X[col] = X[col].str.lower().str.replace(' ', '')
		# if not iterating
		else:
			X[list_cols] = X[list_cols].applymap(lambda x: clean_text(str_text=x))

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return
		return X

# define class for date parts
class DateParts(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, str_datecol='applicationdate', bool_verbose=True, str_message='Date parts'):
		self.str_datecol = str_datecol
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# day of week
		X[f'ENG-{self.str_datecol}-dayofweek'] = X[self.str_datecol].dt.dayofweek

		# day of month
		X[f'ENG-{self.str_datecol}-dayofmonth'] = X[self.str_datecol].dt.day / X[self.str_datecol].dt.daysinmonth

		# month
		X[f'ENG-{self.str_datecol}-month'] = X[self.str_datecol].dt.month

		# quarter
		X[f'ENG-{self.str_datecol}-quarter'] = X[self.str_datecol].dt.quarter

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return
		return X

# define class for cyclic features
class CyclicFeatures(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, str_datecol='applicationdate', bool_verbose=True, list_time_pd=['q_y','m_y','d_w','d_m'], str_message='Cyclic Features'):
		self.str_datecol = str_datecol
		self.bool_verbose = bool_verbose
		self.list_time_pd = list_time_pd
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start timer
		time_start = time.perf_counter()

		# Quarter relative to year
		if 'q_y' in self.list_time_pd:
			# sin
			X[f'ENG-{self.str_datecol}-quarter_year_sin'] = np.sin((X[self.str_datecol].dt.quarter-1) * (2*np.pi/4))
			# cos
			X[f'ENG-{self.str_datecol}-quarter_year_cos'] = np.cos((X[self.str_datecol].dt.quarter-1) * (2*np.pi/4))
			# tan
			X[f'ENG-{self.str_datecol}-quarter_year_tan'] = X[f'ENG-{self.str_datecol}-quarter_year_sin'] / X[f'ENG-{self.str_datecol}-quarter_year_cos']
		else:
			pass
		# Month relative to year
		if 'm_y' in self.list_time_pd:
			# sin
			X[f'ENG-{self.str_datecol}-month_year_sin'] = np.sin((X[self.str_datecol].dt.month-1) * (2*np.pi/12))
			# cos
			X[f'ENG-{self.str_datecol}-month_year_cos'] = np.cos((X[self.str_datecol].dt.month-1) * (2*np.pi/12))
			# tan
			X[f'ENG-{self.str_datecol}-month_year_tan'] = X[f'ENG-{self.str_datecol}-month_year_sin'] / X[f'ENG-{self.str_datecol}-month_year_cos']
		else:
			pass
		# Day relative to week
		if 'd_w' in self.list_time_pd:
			# sin
			X[f'ENG-{self.str_datecol}-day_week_sin'] = np.sin((X[self.str_datecol].dt.dayofweek-1) * (2*np.pi/7))
			# cos
			X[f'ENG-{self.str_datecol}-day_week_cos'] = np.cos((X[self.str_datecol].dt.dayofweek-1) * (2*np.pi/7))
			# tan
			X[f'ENG-{self.str_datecol}-day_week_tan'] = X[f'ENG-{self.str_datecol}-day_week_sin'] / X[f'ENG-{self.str_datecol}-day_week_cos']
		else:
			pass
		# Day relative to month
		if 'd_m' in self.list_time_pd:
			# sin
			X[f'ENG-{self.str_datecol}-day_month_sin'] = np.sin((X[self.str_datecol].dt.day-1) * (2*np.pi/X[self.str_datecol].dt.daysinmonth))
			# cos 
			X[f'ENG-{self.str_datecol}-day_month_cos'] = np.cos((X[self.str_datecol].dt.day-1) * (2*np.pi/X[self.str_datecol].dt.daysinmonth))
			# tan
			X[f'ENG-{self.str_datecol}-day_month_tan'] = X[f'ENG-{self.str_datecol}-day_month_sin'] / X[f'ENG-{self.str_datecol}-day_month_cos']
		else:
			pass
		# Day relative to year
		if 'd_y' in self.list_time_pd:
			# sin
			X[f'ENG-{self.str_datecol}-day_year_sin'] = np.sin((X[self.str_datecol].dt.dayofyear-1) * (2*np.pi/365))
			# cos
			X[f'ENG-{self.str_datecol}-day_year_cos'] = np.cos((X[self.str_datecol].dt.dayofyear-1) * (2*np.pi/365))
			# tan
			X[f'ENG-{self.str_datecol}-day_year_tan'] = X[f'ENG-{self.str_datecol}-day_year_sin'] / X[f'ENG-{self.str_datecol}-day_year_cos']
		else:
			pass

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# return
		return X

# rounding binner
class RoundBinning(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_round, bool_iterate=True, bool_verbose=True, str_message='Binner'):
		self.dict_round = dict_round
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_round = {key: val for key, val in self.dict_round.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_round.items()):
				X[key] = val * round(pd.to_numeric(X[key]) / val)
		# if not iterating
		else:
			# define helper to make lambda function shorter
			def round_it(col, dict_round):
				# get val
				val = self.dict_round[col.name]
				# return
				return val * round(pd.to_numeric(col) / val)
			# apply function
			X[list(dict_round.keys())] = X[list(dict_round.keys())].apply(lambda col: round_it(col=col, dict_round=dict_round), axis=0)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X

# generic feature engineering class
class GenericFeatureEngineering(BaseEstimator, TransformerMixin):
	# init
	def __init__(self, dict_fe, str_datecol='applicationdate__app', bool_verbose=True, str_message='Feature Engineer'):
		self.dict_fe = dict_fe
		self.str_datecol = str_datecol
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# total attempted
		int_attempted = 0
		# total success
		int_success = 0
		# iterate through key val pairs
		for str_key, list_tpl in self.dict_fe.items():
			# iterate through list of tuples
			for tpl in list_tpl:
				# get numerator
				str_numerator = tpl[0]
				# get denominator
				str_denominator = tpl[1]
				# get new col name
				str_new_col = f'ENG-{str_numerator}-{str_key}-{str_denominator}'
				# add attempt
				int_attempted += 1
				# add success
				int_success += 1
				# get series
				try:
					# logic for numerator
					if str_numerator == self.str_datecol:
						# ser numerator
						ser_numerator = X[str_numerator].dt.year
					else:
						# ser numerator
						ser_numerator = X[str_numerator]
					# logic for denominator
					if str_denominator == self.str_datecol:
						# ser denominator
						ser_denominator = X[str_denominator].dt.year
					else:
						# ser denominator
						ser_denominator = X[str_denominator]
				except KeyError:
					# subtract success
					int_success -= 1
					# skip the rest of the loop
					continue
				
				# calculate
				if str_key == 'div':
					X[str_new_col] = ser_numerator / ser_denominator
				elif str_key == 'mult':
					X[str_new_col] = ser_numerator * ser_denominator
				elif str_key == 'add':
					X[str_new_col] = ser_numerator + ser_denominator
				elif str_key == 'sub':
					X[str_new_col] = ser_numerator - ser_denominator

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message} ({int_success}/{int_attempted}): {flt_sec:0.5} sec.')
		# return
		return X

# custom range mapper
class CustomRangeMapper(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_custom_range_map, bool_iterate=True, bool_verbose=True, str_message='Custom Range Mapper'):
		self.dict_custom_range_map = dict_custom_range_map
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# map
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_custom_range_map = {key: val for key, val in self.dict_custom_range_map.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_custom_range_map.items()):
				X[key] = X[key].apply(val)
		# if not iterating
		else:
			X[list(dict_custom_range_map.keys())] = X.apply(dict_custom_range_map)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# imputer
class Imputer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_imputations, bool_iterate=True, bool_verbose=True, str_message='Imputer'):
		self.dict_imputations = dict_imputations
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# fillna
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_imputations = {key: val for key, val in self.dict_imputations.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_imputations.items()):
				X[key] = X[key].fillna(val)
		# if not iterating
		else:
			X.fillna(dict_imputations, inplace=True)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# define value replacer class
class FeatureValueReplacer(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, dict_value_replace, bool_iterate=True, bool_verbose=True, str_message='Value Replacer'):
		self.dict_value_replace = dict_value_replace
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# rm key val pairs not in X
		dict_value_replace = {key: val for key, val in self.dict_value_replace.items() if key in list(X.columns)}
		# if iterating
		if self.bool_iterate:
			for key, val in tqdm (dict_value_replace.items()):
				X[key] = X[key].replace(val)
		# if not iterating
		else:
			X.replace(self.dict_value_replace, inplace=True)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# replace inf and -inf with NaN
class ReplaceInf(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True, str_message='Inf Replacer'):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]

		# if iterating
		if self.bool_iterate:
			for col in tqdm (list_cols):
				X[col] = X[col].replace([np.inf, -np.inf], np.nan)
		# if not iterating
		else:
			X[list_cols] = X[list_cols].replace([np.inf, -np.inf], np.nan)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X

# clip values
class ClipValues(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols, bool_iterate=True, bool_verbose=True, str_message='Value clipper', int_clip=0):
		self.list_cols = list_cols
		self.bool_iterate = bool_iterate
		self.bool_verbose = bool_verbose
		self.str_message = str_message
		self.int_clip = int_clip
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# future proof
		list_cols = [col for col in self.list_cols if col in list(X.columns)]

		# if iterating
		if self.bool_iterate:
			for col in tqdm (list_cols):
				X[col] = np.clip(a=X[col], a_min=self.int_clip, a_max=None)
		else:
			X[list_cols] = np.clip(a=X[list_cols], a_min=self.int_clip, a_max=None)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		if self.bool_verbose:
			print(f'{self.str_message}: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X

# define preprocessing model class
class PreprocessingModel(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_transformers):
		self.list_transformers = list_transformers
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# iterate through transformers
		for transformer in self.list_transformers:
			try:
				X = transformer.transform(X)
			except KeyError:
				pass

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'Preprocessing Model: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# --------------------------------------------------------------------------------
# --------------------------------------------------------------------------------

# define string converter
class StringConverter(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self, list_cols):
		self.list_cols = list_cols
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# make sure all cols are in X
		list_cols = [col for col in self.list_cols if col in list(X.columns)]
		# convert to string
		X[list_cols] = X[list_cols].applymap(str)

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'String Converter: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return
		return X

# new amt financed
class NewAmountFinanced(BaseEstimator, TransformerMixin):
	# initialize
	def __init__(self):
		pass
	# fit
	def fit(self, X, y=None):
		return self
	# transform
	def transform(self, X):
		# start time
		time_start = time.perf_counter()

		# calculate
		X['ENG-fltamountfinanced__app'] = X['fltamountfinanced__app'] - X['fltgapinsurance__app'] - X['fltservicecontract__app']

		# end time
		time_end = time.perf_counter()
		# flt_sec
		flt_sec = time_end - time_start
		# print
		print(f'New Amount Financed: {flt_sec:0.5} sec.')
		# save to object
		self.flt_sec = flt_sec
		# return X
		return X