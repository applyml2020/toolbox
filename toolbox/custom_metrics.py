# custom metrics
import numpy as np
from sklearn.metrics import f1_score
from scipy.special import expit

# define class for precision-recall AUC (catboost)
class F1Threshold:	
	# Returns whether great values of metric error are better
	def is_max_optimal(self):
		return True
	# Compute metric
	def evaluate(self, approxes, target, weight):
		# make sure theres only 1 item in approxes
		assert len(approxes) == 1
		# make sure there are as many actual (target) as there are predictions (approxes[0])
		assert len(target) == len(approxes[0])
		# set target to integer and save as y_true
		y_true = np.array(target).astype(int)
		# get predictions, fit to logistic sigmoid function, and set as float
		y_pred = expit(approxes[0]).astype(float)
		# cutoff at threshold
		y_pred = np.where(np.array(y_pred) < 0.42, 0, 1)
		# generate f1
		error = f1_score(y_true=y_true, y_pred=y_pred)
		# return
		return error, 1
	# get final error   
	def get_final_error(self, error, weight):
		# Returns final value of metric
		return error
