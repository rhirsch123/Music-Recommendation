from hdf5_getters import *
import sqlite3
import numpy as np


path = 'datasets/track_metadata.db'
conn = sqlite3.connect(path)

# add rows
c = conn.cursor()

c.execute("ALTER TABLE songs ADD COLUMN loudness real")
c.execute("ALTER TABLE songs ADD COLUMN tempo real")
c.execute("ALTER TABLE songs ADD COLUMN mode int")
conn.commit()

path = 'datasets/msd_summary_file.h5'
sum_file = open_h5_file_read(path)

file_count = 0
for idx in range(0, 1000000):
  track_id = get_track_id(sum_file, idx).decode('utf-8')
  loudness = get_loudness(sum_file, idx)
  loudness = str(loudness) if not np.isnan(loudness) else "-1"
  tempo = get_tempo(sum_file, idx)
  tempo = str(tempo) if not np.isnan(tempo) else "-1"

  c.execute(f"UPDATE songs SET loudness={loudness}, tempo={tempo} WHERE track_id='{track_id}'")

  mode = get_mode(sum_file, idx)

  c.execute(f"UPDATE songs SET mode={mode} WHERE track_id='{track_id}'")

  file_count += 1
  if file_count % 200 == 0:
    conn.commit()

sum_file.close()
conn.commit()
conn.close()

print(f'added data of {file_count} tracks')
