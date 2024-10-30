import pickle
import os

# save object to pickle file
def save_object(obj, filename, directory='pickle_storage'):
	file_path = os.path.join(directory, filename)
	with open(file_path, 'wb') as f:
		pickle.dump(obj, f)


# load object from pickle file
def load_object(filename, directory='pickle_storage'):
	file_path = os.path.join(directory, filename)
	with open(file_path, 'rb') as f:
		return pickle.load(f)
