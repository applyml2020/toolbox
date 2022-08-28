# general
import os
import pickle

# define class
class General:
	# initialize
	def __init__(self, str_dirname_output='./output'):
		self.str_dirname_output = str_dirname_output
	# create directory
	def create_directory(self):
		# create output dir
		try:
			os.mkdir(f'{self.str_dirname_output}')
			print(f'Created directory {self.str_dirname_output}')
		except FileExistsError:
			print(f'Directory {self.str_dirname_output} already exists')
		# return object
		return self
	# pickle to file
	def pickle_to_file(self, item_to_pickle, str_filename='cls_eda.pkl'):
		# save
		pickle.dump(item_to_pickle, open(f'{self.str_dirname_output}/{str_filename}', 'wb'))
		# return object
		return self

"""
# Example:

# initialize class
cls_general = General(
	str_dirname_output='./opt/ml/model',
)

# create directory
cls_general.create_directory()

# pickle class
cls_general.pickle_to_file(
	item_to_pickle=cls_general, 
	str_filename='cls_eda.pkl',
)
"""