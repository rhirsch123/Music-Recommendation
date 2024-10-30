from pickle_utils import *
import sqlite3
import numpy as np

path = 'datasets/artist_term.db'
term_conn = sqlite3.connect(path)

artist_ids = []
for artist in term_conn.execute("SELECT artist_id FROM artists").fetchall():
	artist_ids.append(artist[0])

# combined list of artist terms and tags
term_list = []
for term in term_conn.execute("SELECT term FROM terms").fetchall():
	term_list.append(term[0])

for tag in term_conn.execute("SELECT mbtag FROM mbtags").fetchall():
	term_list.append(tag[0])

# make unique
term_list = list(set(term_list))


matrix = []
for artist_id in artist_ids:
	term_vector = np.zeros(len(term_list), dtype=int)

	current_artist_terms = []

	q = f"SELECT term FROM artist_term WHERE artist_id='{artist_id}'"
	for term in term_conn.execute(q).fetchall():
		current_artist_terms.append(term[0])
	
	q = f"SELECT mbtag FROM artist_mbtag WHERE artist_id='{artist_id}'"
	for tag in term_conn.execute(q).fetchall():
		current_artist_terms.append(tag[0])

	for i in range(len(term_list)):
		if term_list[i] in current_artist_terms:
			term_vector[i] = 1
	
	matrix.append(term_vector)

matrix = np.array(matrix)

save_object(matrix, 'terms_matrix.pkl')
save_object(artist_ids, 'artist_ids.pkl')
