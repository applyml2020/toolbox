# feature selection
from .drift_detection import DriftDetection
import numpy as np
import sklearn.metrics as skm
import pickle
from tqdm import tqdm
import pandas as pd 

# define class
class FeatureSelection(DriftDetection):
	# initialize
	def __init__(self, cls_model_preprocessing, str_dirname_output='/opt/ml/model', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		DriftDetection.__init__(self, cls_model_preprocessing, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.cls_model_preprocessing = cls_model_preprocessing
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# get performance difference
	def get_performance_difference(self, df_test, flt_prop_rows=0.10):
		# get inference model
		try:
			cls_model_inference = self.dict_pipeline['model_inference']
		except AttributeError:
			cls_model_inference = self.cls_model_inference
		# create list for subsetting
		list_subset = cls_model_inference.feature_names_
		list_subset.append(self.str_target)
		# subset to features we need
		df_test = df_test[list_subset]
		# get the string eval metric
		str_eval_metric = cls_model_inference.get_params()['eval_metric']
		# get number of rows for sample
		int_nrows = int(round(df_test.shape[0] * flt_prop_rows))
		# randomly sample rows from df_test
		df_test = df_test.sample(
			n=int_nrows, 
    		replace=False, 
    		random_state=42,
		)
		# generate eval metric
		if str_eval_metric == 'AUC':
			# predict
			y_hat = cls_model_inference.predict_proba(df_test[cls_model_inference.feature_names_])[:,1]
			# eval metric
			flt_eval_metric = skm.roc_auc_score(y_true=df_test[self.str_target], y_score=y_hat)
		
		# iterate through features in the model
		dict_diff = {}
		for col in tqdm (cls_model_inference.feature_names_):
			# save series so we can replace it later
			ser_col_old = df_test[col]
			# shuffle, save as list in case of index reordering, and assign
			df_test[col] = list(df_test[col].sample(frac=1, random_state=42).reset_index(drop=True))
			# generate predictions
			if str_eval_metric == 'AUC':
				y_hat = cls_model_inference.predict_proba(df_test[cls_model_inference.feature_names_])[:,1]
				flt_eval_metric_new = skm.roc_auc_score(y_true=df_test[self.str_target], y_score=y_hat)
				flt_diff = flt_eval_metric - flt_eval_metric_new
			# put into dict_diff
			dict_diff[col] = flt_diff
			# replace col
			df_test[col] = ser_col_old
		# save dict_diff
		pickle.dump(dict_diff, open(f'{self.str_dirname_output}/dict_diff.pkl', 'wb'))
		# save to object
		self.dict_diff = dict_diff
		# return object
		return self
	# iterative feature selection
	def iterative_feature_selection(self, df_train, df_test, list_cols_model, dict_monotone_constraints, list_feat_force, list_class_weights, int_n_iterations=1000, str_eval_metric='AUC', flt_learning_rate=None, int_random_state=42):
		# make sure the list_feat_force in list_cols_model
		list_cols_model = list_cols_model + list_feat_force
		list_cols_model = list(dict.fromkeys(list_cols_model))
		# set int_n_imp_zero to 1
		int_n_imp_zero = 1
		# while int_n_imp_zero > 0, fit models
		list_dict_row = []
		int_counter = 1
		while int_n_imp_zero > 0:
			# fit model
			self.fit_model(
				df_train=df_train, 
				df_test=df_test, 
				list_cols_model=list_cols_model, 
				dict_monotone_constraints=dict_monotone_constraints, 
				int_n_iterations=int_n_iterations, 
				str_eval_metric=str_eval_metric, 
				flt_learning_rate=flt_learning_rate, 
				list_class_weights=list_class_weights, 
				int_random_state=int_random_state,
			)
			# create feat imp df
			df_feat_imp = pd.DataFrame({
				'feature': self.cls_model_inference.feature_names_, 
				'importance': self.cls_model_inference.feature_importances_,
			})
			# sort descending
			df_feat_imp.sort_values(by='importance', ascending=False, inplace=True)
			# get eval metric
			flt_eval_metric = self.dict_pipeline['flt_eval_metric']
			# row
			dict_row = {'iteration': int_counter, 'n_feats': df_feat_imp.shape[0], 'eval_metric': flt_eval_metric}
			# append
			list_dict_row.append(dict_row)
			# create df
			df_iterative_feat_select = pd.DataFrame(list_dict_row)
			# save
			df_iterative_feat_select.to_csv(f'{self.str_dirname_output}/df_iterative_feat_select.csv', index=False)
			# save to object
			self.df_iterative_feat_select = df_iterative_feat_select
			# get list_cols_model
			list_cols_model = list(df_feat_imp[df_feat_imp['importance']>0]['feature'])
			# make sure forced features are in list_cols_model
			for col in list_feat_force:
				if col not in list_cols_model:
					list_cols_model.append(col)
			# create df
			df_cols_model = pd.DataFrame({'feature': list_cols_model})
			# save
			df_cols_model.to_csv(f'{self.str_dirname_output}/df_cols_model.csv', index=False)
			# save to object
			self.df_cols_model = df_cols_model
			# get 0 imp feats
			list_feats_zero_imp = list(df_feat_imp[df_feat_imp['importance']==0]['feature'])
			# rm list_feat_force and check for number of 0 imp feats (because we dont care if a forced feature has 0 importance)
			int_n_imp_zero = len([col for col in list_feats_zero_imp if col not in list_feat_force])
			# increase counter
			int_counter += 1
		# return object
		return self

"""
	# feature selection
	def get_average_feature_importance(self, df_train, df_valid, dict_monotone_constraints, int_n_models=10, int_n_iterations=1000, str_eval_metric='AUC'):
		# list cols
		list_cols = [col for col in df_train.columns if col != self.str_target]
		# get nonnumeric cols
		list_non_numeric = [col for col in list_cols if not is_numeric_dtype(df_train[col])]
		# pool
		pool_train = cb.Pool(df_train[list_cols], df_train[self.str_target], cat_features=list_non_numeric)
		pool_valid = cb.Pool(df_valid[list_cols], df_valid[self.str_target], cat_features=list_non_numeric)

		# empty list
		list_dict_feat_imp = []
		# iterate through random states
		for a in range(int_n_models):
			# message
			print('')
			print(f'Model {a+1}/{int_n_models}')
			
			# create dictionary
			dict_hyperparameters = {
				'task_type': 'CPU',
				'nan_mode': 'Min',
				'iterations': int_n_iterations,
				'eval_metric': str_eval_metric,
				'monotone_constraints': dict_monotone_constraints,
				'random_state': a+1,
			}

			# if doing classification
			if self.bool_target_binary:
				# put into dict_hyperparameters
				dict_hyperparameters['class_weights'] = list(compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train))
				# initialize model
				cls_model = cb.CatBoostClassifier(**dict_hyperparameters)
			# if doing regression
			else:
				# initialize model
				cls_model = cb.CatBoostRegressor(**dict_hyperparameters)

			# fit
			cls_model.fit(
				pool_train, 
				eval_set=[pool_valid], 
				use_best_model=True, 
				early_stopping_rounds=int(round(int_n_iterations*0.20)), 
				verbose=100,
			)

			# get feat imp dict
			dict_feat_imp = dict(zip(cls_model.feature_names_, cls_model.feature_importances_))
			# append to list_dict_feat_imp
			list_dict_feat_imp.append(dict_feat_imp)
			# create df
			df_output = pd.DataFrame(list_dict_feat_imp)

		# get mean by column
		ser_feat_imp = df_output.apply(lambda col: col.mean(), axis=0).sort_values(ascending=False)
		# create df
		df_imp = pd.DataFrame({'feature': ser_feat_imp.index, 'importance': ser_feat_imp.values})
		# get cumsum
		df_imp['importance_cumsum'] = np.cumsum(df_imp['importance'])
		# save
		df_imp.to_csv(f'{self.str_dirname_output}/df_imp.csv', index=False)
		# return object
		return self
"""