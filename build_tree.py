from sklearn.neighbors import BallTree
from pickle_utils import *

matrix = load_object('terms_matrix.pkl')

tree = BallTree(matrix)

save_object(tree, 'artist_terms_tree.pkl')

