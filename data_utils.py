import json
import pandas as pd
import os

# Converts MPD JSON into CSV format with user_id, track_id, timestamp
# Each playlist is treated as a user, and each track gets a generated timestamp

def parse_mpd_to_csv(json_file, output_csv):
    with open(json_file, 'r') as f:
        mpd_data = json.load(f)

    rows = []
    for playlist in mpd_data['playlists']:
        pid = playlist['pid']  # Treat playlist ID as user ID
        base_time = 1600000000  # Arbitrary start time for sequencing
        for i, track in enumerate(playlist['tracks']):
            track_uri = track['track_uri']
            # Convert track URI string into a numeric ID (hash mapped)
            track_id = abs(hash(track_uri)) % 100000
            timestamp = base_time + i * 60  # Simulate timestamp with spacing
            rows.append([pid, track_id, timestamp])

    # Create and save DataFrame
    df = pd.DataFrame(rows, columns=["user_id", "track_id", "timestamp"])
    df.to_csv(output_csv, index=False)
