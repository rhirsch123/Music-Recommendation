import pandas as pd
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix, coo_matrix
from pickle_utils import *

# user	item	play_count
df = pd.read_csv('datasets/train_triplets.txt',
		sep='\t', names=['user', 'item', 'interaction'], header=None)


# map id's to index
user_to_index = {user: idx for idx, user in enumerate(df['user'].unique())}
item_to_index = {item: idx for idx, item in enumerate(df['item'].unique())}

df['user'] = df['user'].map(user_to_index)
df['item'] = df['item'].map(item_to_index)


# als model must take csr matrix
row = df['user'].values
col = df['item'].values
data = df['interaction'].values

sparse_user_item = coo_matrix((data, (row, col))).tocsr()

model = AlternatingLeastSquares()

model.fit(sparse_user_item)

save_object(model, 'als_model.pkl')
save_object(item_to_index, 'song_id_to_index.pkl')
