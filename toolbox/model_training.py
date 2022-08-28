# model training
from .feature_selection import FeatureSelection
import numpy as np

# define class for model training
class ModelTraining(FeatureSelection):
	# initialize
	def __init__(self, cls_model_preprocessing, str_dirname_output='/opt/ml/model', str_target='target', str_datecol='date', bool_target_binary=True):
		# initialize parent class
		FeatureSelection.__init__(self, cls_model_preprocessing, str_dirname_output, str_target, str_datecol, bool_target_binary)
		# save arguments to object
		self.cls_model_preprocessing = cls_model_preprocessing
		self.str_dirname_output = str_dirname_output
		self.str_target = str_target
		self.str_datecol = str_datecol
		self.bool_target_binary = bool_target_binary
	# get features
	def get_features(self, df_imp, int_n_feats=1000, list_feat_force=[]):
		# subset and get list of feats
		list_cols_model = list(df_imp[df_imp['rank']<=int_n_feats]['feature'])
		# extend list
		list_cols_model.extend(list_feat_force)
		# rm dups
		list_cols_model = list(dict.fromkeys(list_cols_model))
		# return 
		return list_cols_model