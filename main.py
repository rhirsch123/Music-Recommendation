import sqlite3
from pickle_utils import *
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# ball tree for artist terms
terms_tree = load_object('artist_terms_tree.pkl')

# list of all artist_id's corresponding to rows in term matrix
artist_ids = load_object('artist_ids.pkl')

# matrix of all artist terms
terms_matrix = load_object('terms_matrix.pkl')

# als model for collaborative filter
model = load_object('als_model.pkl')

# mapping used for als model
song_id_to_index = load_object('song_id_to_index.pkl')


# class to store data of a song
class Song:
	def __init__(self, track_id, song_id, title, artist_name, release,
		artist_id, artist_familiarity, duration, loudness, tempo):
		''' track_id is slightly different than song_id. Two songs can have the
	  same song_id. Need song_id for collaborative filter '''

		self.track_id = track_id
		self.song_id = song_id
		self.title = title
		self.artist_name = artist_name
		self.release = release
		self.artist_id = artist_id
		self.artist_familiarity = artist_familiarity
		self.duration = duration
		self.loudness = loudness
		self.tempo = tempo
	
	def __str__(self):
		return f'{self.title} by {self.artist_name}'


# dictionary of all playlists in form title: list of song objects
playlists = load_object('playlists.pkl')


# function to sort search results by
def sort_results_function(song, search):
	familiarity = song.artist_familiarity

	if search == song.title.lower():
		return (5, familiarity)
	if search == song.artist_name.lower():
		return (4, familiarity)
	if search == song.release.lower():
		return (3, familiarity)
	if search in song.title.lower():
		return (2, familiarity)
	if search in song.artist_name.lower():
		return (1, familiarity)
	return (0, familiarity)


conn = sqlite3.connect('datasets/track_metadata.db')

# function to sort artist list by
def get_artist_familiarity(artist_id, conn=conn):
	q = f"SELECT artist_familiarity FROM songs WHERE artist_id='{artist_id}' LIMIT 1"
	return conn.execute(q).fetchall()[0][0]


# user search for songs to select
def search_for_song():
	while True:
		search = input('Search for song: ').lower()

		results = []

		q = "SELECT track_id, song_id, title, artist_name, release, artist_id, artist_familiarity, duration, loudness, tempo FROM songs"

		for song in conn.execute(q).fetchall():
			combined = f'{song[2]} {song[3]} {song[4]}'
			if search in combined.lower():
				results.append(Song(song[0], song[1], song[2], song[3], song[4],
					song[5], song[6], song[7], song[8], song[9]))
	
		results.sort(key=lambda song: sort_results_function(song, search), reverse=True)

		print('Enter number of song to add or prompt:\n')
		for i in range(len(results[:20])):
			print(f'{i}. {results[i]}')
	
		print('\n20. Search again')
		print('21. Cancel')
		index = int(input())
	
		if index >= 0 and index <= 19:
			return results[index]

		if index == 21:
			return None


# generates list of recommendations from playlist based on content data
def get_recommendations_content(playlist):
	query_artists = [song.artist_id for song in playlist]

	# get similar artists based on terms
	term_query_matrix = terms_matrix[artist_ids.index(query_artists[0])]
	query_mean = term_query_matrix

	for i in range(1, len(playlist)):
		term_query_matrix = np.vstack((term_query_matrix,
			terms_matrix[artist_ids.index(query_artists[i])]))

	
	if len(playlist) > 1:
		query_mean = np.mean(term_query_matrix, axis=0)

	distances, indices = terms_tree.query([query_mean], k=100)

	result_ids = []
	for i in indices[0]:
		result_ids.append(artist_ids[i])

	# take the most popular artists of the best matches
	result_ids.sort(key=get_artist_familiarity, reverse=True)
	result_ids = result_ids[:25]

	# make tree of songs from selected artists
	song_list = []
	for artist_id in result_ids:
		q = f"SELECT track_id, song_id, title, artist_name, release, artist_id, artist_familiarity, duration, loudness, tempo FROM songs WHERE artist_id='{artist_id}'"

		for song in conn.execute(q).fetchall():
			song_list.append(Song(song[0], song[1], song[2], song[3], song[4],
				song[5], song[6], song[7], song[8], song[9]))

	song_matrix = []
	for song in song_list:
		#song_matrix.append([song.duration, song.loudness, song.tempo])
		song_matrix.append([song.loudness, song.tempo])

	song_matrix = np.array(song_matrix)

	knn_model = NearestNeighbors(metric='euclidean', algorithm='brute')
	''' experimented with other metrics; something like cosine doesn't care
	about magnitude '''

	knn_model.fit(song_matrix)

	# get most similar songs
	song_query_matrix = []
	for song in playlist:
		#song_query_matrix.append([song.duration, song.loudness, song.tempo])
		song_query_matrix.append([song.loudness, song.tempo])

	query_mean = song_query_matrix[0]
	if len(playlist) > 1:
		query_mean = np.mean(song_query_matrix, axis=0)

	distances, indices = knn_model.kneighbors([query_mean],
		n_neighbors=len(song_list))

	# maximum of 2 songs per artist
	artist_use_count = {}
	for artist in result_ids:
		artist_use_count[artist] = 0

	# take top N songs
	N = 20
	recommendations = []
	song_count = 0
	for i in indices[0]:
		if song_count >= N:
			break

		rec = song_list[i]
		if rec not in playlist and artist_use_count[rec.artist_id] < 2:
			recommendations.append(rec)
			artist_use_count[rec.artist_id] += 1
			song_count += 1

	return recommendations


''' generates list of recommendations from playlist based on als
collaborative filter '''
def get_recommendations_collaborative(playlist):
	# user interaction vector for all songs
	user_vector = np.zeros(len(song_id_to_index))
	for song in playlist:
		if song.song_id in song_id_to_index:
			# somewhat arbitrary choice of 5 to represent liked song playcount
			user_vector[song_id_to_index[song.song_id]] = 5
	
	user_vector = csr_matrix(user_vector)

	item_factors = model.item_factors

	# user's latent factors
	user_latent = user_vector.dot(item_factors)

	# calculate scores for all items
	scores = user_latent.dot(item_factors.T)

	# take top N scored songs
	N = 20
	top_items = np.argsort(-scores[0])[:N]

	# convert to list of song objects
	index_to_song_id = {idx: item for item, idx in song_id_to_index.items()}
	recommended_song_ids = [index_to_song_id[idx] for idx in top_items]
	
	recommendations = []
	MIN_SCORE = 0.5
	for song_id in recommended_song_ids:
		score = scores[0][song_id_to_index[song_id]]
		if score >= 0.5:
			q = f"SELECT track_id, song_id, title, artist_name, release, artist_id, artist_familiarity, duration, loudness, tempo FROM songs WHERE song_id='{song_id}' LIMIT 1"
			result = conn.execute(q).fetchall()[0]
			song = Song(result[0], result[1], result[2], result[3], result[4],
				result[5], result[6], result[7], result[8], result[9])
			recommendations.append(song)
	
	return recommendations



# user flow
while True:
	print('1. Create playlist')
	print('2. Edit playlist')
	print('3. Done')
	task = input()

	if task == '3':
		break

	if task == '1':
		print('1. Back')
		playlist_name = input('Enter playlist name: ')
		if playlist_name != '1':
			playlists[playlist_name] = []

	if task == '2':
		print('Choose playlist:')
		for key in playlists:
			print(key)

		playlist_name = input()
		playlist = playlists[playlist_name]

		for song in playlist:
			print(song)
		print()
		
		while True:
			print('1. Add song')
			print('2. Remove song')
			print('3. Delete playlist')
			print('4. Done')
			edit_task = input()

			if edit_task == '4':
				break

			if edit_task == '1':
				print('1. Search for song')

				if len(playlist) > 0:
					print('2. Generate Recommendations')

				add_task = input()

				if add_task == '1':
					new_song = search_for_song()
					if new_song:
						playlist.append(new_song)
				
				if add_task == '2':
					content_recs = get_recommendations_content(playlist)
					collab_recs = get_recommendations_collaborative(playlist)

					idx = 0
					print('Similar songs:')
					for song in content_recs:
						print(f'{idx}. {song}')
						idx += 1
					
					if len(collab_recs) > 0:
						print('\nSimilar users also liked:')
						for song in collab_recs:
							print(f'{idx}. {song}')
							idx += 1

					print(f'\n{idx}. Back')

					
					add_index = int(input())
					if add_index < len(content_recs):
						playlist.append(content_recs[add_index])
					elif add_index < len(content_recs) + len(collab_recs):
						playlist.append(collab_recs[add_index - len(content_recs)])

			if edit_task == '2':
				for i in range(len(playlist)):
					print(f'{i}. {playlist[i]}')

				print(f'\n{len(playlist)}. Back')
				
				remove_index = int(input())
				if remove_index < len(playlist):
					del playlist[remove_index]


			if edit_task == '3':
				del playlists[playlist_name]
				break


conn.close()
save_object(playlists, 'playlists.pkl')
