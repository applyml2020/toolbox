# preprocessing
from .train_valid_test_split import TrainValidTestSplit
from pandas.api.types import is_numeric_dtype
import catboost as cb
import sklearn.metrics as skm
import pickle
import numpy as np
import pandas as pd

# define class
class LeakyFeatures(TrainValidTestSplit):
	# initialize
	def __init__(self, str_dirname_output='./output', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		TrainValidTestSplit.__init__(self, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# cehck for ID columns
	def check_for_id_cols(self, list_cols):
		# check for id
		list_id_cols = [col for col in list_cols if 'id' in col.lower()]
		# save to object
		self.list_id_cols = list_id_cols
		# return object
		return self
	# check for date columns
	def check_for_date_columns(self, list_cols):
		# check for date
		list_date_cols = []
		for col in list_cols:
			# check for date
			if ('date' in col.lower()) or ('dtm' in col.lower()) or ('dte' in col.lower()):
				list_date_cols.append(col)
			else:
				pass
		# save to object
		self.list_date_cols = list_date_cols
		# return object
		return self
	# check for score columns
	def check_for_score_columns(self, list_cols):
		# check for score
		list_score_cols = [col for col in list_cols if 'score' in col.lower()]
		# save to object
		self.list_score_cols = list_score_cols
		# return object
		return self
	# fit model
	def fit_model(self, df_train, df_test, list_cols_model, verbose=100, dict_monotone_constraints=None, int_n_iterations=1000, str_eval_metric='AUC', flt_learning_rate=None, list_class_weights=None, int_random_state=42):
		# get the nonnumeric features
		list_non_numeric = [col for col in list_cols_model if not is_numeric_dtype(df_train[col])]
		# save test data for eval later
		y_test = df_test[self.str_target]
		df_test = df_test[list_cols_model]
		# pool
		y_train = df_train[self.str_target] # save for balancing target
		pool_train = cb.Pool(df_train[list_cols_model], df_train[self.str_target], cat_features=list_non_numeric)
		pool_test = cb.Pool(df_test, y_test, cat_features=list_non_numeric)
		# create monotone constraints dictionary
		try:
			dict_monotone_constraints = {key: val for key, val in dict_monotone_constraints.items() if key in list_cols_model}
		except AttributeError:
			pass

		# create dictionary
		dict_hyperparameters = {
			'task_type': 'CPU',
			'nan_mode': 'Min',
			'random_state': int_random_state,
			'eval_metric': str_eval_metric,
			'iterations': int_n_iterations,
			'learning_rate': flt_learning_rate,
			'monotone_constraints': dict_monotone_constraints,
		}

		# if doing classification
		if self.bool_target_binary:
			# add class weights
			dict_hyperparameters['class_weights'] = list_class_weights
			# initialize model
			cls_model_inference = cb.CatBoostClassifier(**dict_hyperparameters)
		# if doing regression
		else:
			# initialize model
			cls_model_inference = cb.CatBoostRegressor(**dict_hyperparameters)

		# fit model
		cls_model_inference.fit(
			pool_train, 
			eval_set=[pool_test], 
			verbose=verbose, 
			use_best_model=True, 
			early_stopping_rounds=int(round(int_n_iterations*0.20)),
		)
		# save to object
		self.cls_model_inference = cls_model_inference

		# if doing classification
		if self.bool_target_binary:
			# class
			y_hat_class = cls_model_inference.predict(df_test[cls_model_inference.feature_names_])
			# probability
			y_hat_proba = cls_model_inference.predict_proba(df_test[cls_model_inference.feature_names_])[:,1]
			# if eval metric is AUC
			if str_eval_metric == 'AUC':
				# eval metric
				flt_eval_metric = skm.roc_auc_score(y_true=y_test, y_score=y_hat_proba)
			# if eval metric is F1
			elif str_eval_metric == 'F1':
				# eval metreic
				flt_eval_metric = skm.f1_score(y_true=y_test, y_pred=y_hat_class)
		# if doing regression
		else:
			# value
			y_hat = cls_model_inference.predict(df_test[cls_model_inference.feature_names_])
			# if eval metric is RMSE
			if str_eval_metric == 'RMSE':
				# eval metric
				flt_eval_metric = np.sqrt(skm.mean_squared_error(y_true=y_test, y_pred=y_hat))

		# get preprocessing model
		try:
			cls_model_preprocessing = self.cls_model_preprocessing
		except AttributeError:
			cls_model_preprocessing = None

		# create dictionary
		dict_pipeline = {
			'list_non_numeric': list_non_numeric,
			'model_preprocessing': cls_model_preprocessing,
			'model_inference': cls_model_inference,
			'str_eval_metric': str_eval_metric,
			'flt_eval_metric': flt_eval_metric,
		}
		# save dict_pipeline
		pickle.dump(dict_pipeline, open(f'{self.str_dirname_output}/dict_pipeline.pkl', 'wb'))
		# save to object
		self.dict_pipeline = dict_pipeline
		# return object
		return self
	# get feature importance
	def get_feature_importance(self):
		# create df
		df_feat_imp = pd.DataFrame({'feature': self.cls_model_inference.feature_names_, 'importance': self.cls_model_inference.feature_importances_})
		# sort descending
		df_feat_imp.sort_values(by='importance', ascending=False, inplace=True)
		# save to .csv
		df_feat_imp.to_csv(f'{self.str_dirname_output}/df_feat_imp.csv', index=False)
		# save to object
		self.df_feat_imp = df_feat_imp
		# return object
		return self