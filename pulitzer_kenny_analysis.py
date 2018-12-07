import lyricsgenius
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
import spotipy
import string

from adjustText import adjust_text
from nltk.corpus import stopwords
from spotipy.oauth2 import SpotifyClientCredentials

# API setup
# NOTE: this requires setup of Spotify and Genius developer accounts for credentials
client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
spotify = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
genius = lyricsgenius.Genius(genius_secret)

# Set up stop words and punctuation for removal
filterWords = stopwords.words("english")
translator = str.maketrans('', '', string.punctuation)


# Returns dataframe of lyrics for tracks in song_list via Genius API
def get_lyrics_from_song_names(artist_name, song_list):
    artist = genius.search_artist(artist_name, max_songs=0)
    lyrics = []
    song_names = []
    
    # Parse songs
    for song_title in song_list:
        song = genius.search_song(song_title, artist.name, take_first_result=True, remove_section_headers=True)
        if song is not None:
            song_lyrics = ' '.join([word for word in song.lyrics.split() if word.lower() not in filterWords])
            lyrics.append(song_lyrics.translate(translator))
            song_names.append(song_title)

    return pd.DataFrame(data=lyrics, index=song_names, columns = ['lyrics'])


# Returns dataframe of track information for tracks from each album in album_list via Spotify API
def get_tracks_from_albums(album_list):
    features_list = []
    track_names = []
    
    # Parse albums
    for album in album_list:
        results = spotify.search(q=album, type='album')
        uri = results['albums']['items'][0]['id']
        album_tracks = spotify.album_tracks(uri)
        
        # Parse tracks
        for item in album_tracks['items']:
            # Handle bonus tracks
            if 'Bonus' not in item['name']:
                audio_features = spotify.audio_features(item['id'])[0]
                audio_features['album'] = album
                features_list.append(audio_features)
                name = item['name'].replace('’', '\'')
                # Replace for asterisk in F*ck Your Ethnicity
                name = name.replace('*', 'u')
                # Handle extended versions
                if 'Extended' in name:
                    name = name.split('Extended')[0]
                if 'FEAT.' in name:
                    name = name.split('FEAT.')[0]
                track_names.append(name)
     
    return pd.DataFrame(data=features_list, index=track_names)


# Returns dictionary mapping each word to a list of corresponding sentiments
def read_lexicon(filename):
    word_sentiments = {}
    
    # Parse lexicon file
    with open(filename, 'r') as f:
        for line in f:
            # Ignore blank lines
            if line.strip():
                word, sentiment, value = line.split()
                if value == '1':
                    # Add sentiment to dictionary
                    cur_list = word_sentiments.get(word, [])
                    cur_list.append(sentiment)
                    word_sentiments[word] = cur_list
    
    return word_sentiments  


# Returns dataframe of sentiment percentages for each track in an input dataframe
def get_sentiment_percentages(track_df, sentiment):
    sentiment_pcts = []
    
    # Parse tracks
    for track_name in track_df.index.values:
        num_sentiment_words = 0
        all_words = track_df.loc[track_name]['lyrics'].split()
        
        # Parse words in lyrics
        for word in all_words:
            if sentiment in word_sentiments_dict.get(word, []):
                num_sentiment_words += 1
                
        sentiment_pcts.append(num_sentiment_words/len(all_words))

    return pd.DataFrame(data=sentiment_pcts, index=track_df.index.values, columns = [sentiment + '_pct'])


# Initialize sentiment dictionary and track dataframe
word_sentiments_dict = read_lexicon('nrc-emotion-lexicon.txt')
spotify_df = get_tracks_from_albums(['Section 80', 'good kid, m.A.A.d city', 'To Pimp a Butterfly', 'DAMN'])
genius_df = get_lyrics_from_song_names('Kendrick Lamar', spotify_df.index.values)
track_df = spotify_df.join(genius_df).dropna(subset=['lyrics'])


# Generate sentiment percentages based on NRC Lexicon sentiments
sentiments = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'negative', 'positive', 'sadness', 'surprise', 'trust']
for sentiment in sentiments:
    sentiment_df = get_sentiment_percentages(track_df, sentiment)
    track_df = track_df.join(sentiment_df)


# Calculate lyrical density and gloom index of tracks
track_df['lyrical_density'] = track_df.apply(lambda row: len(row.lyrics.split()) / row.duration_ms * 1000, axis=1)
track_df['gloom_index'] = track_df.apply(lambda row: 1 - (row.valence) + row.sadness_pct * (row.lyrical_density+1), axis=1)


# Rescale track_df such that gloom indices are in range [0, 100]
rescale_df = track_df
max_gloom_value = track_df['gloom_index'].max()
min_gloom_value = track_df['gloom_index'].min()
rescale_df['gloom_index'] = (track_df['gloom_index'] - min_gloom_value) / (max_gloom_value - min_gloom_value) * 100

rescale_df['gloom_index'].sort_values(ascending=False)


# Data plot

album_list = ['Section 80', 'good kid, m.A.A.d city', 'To Pimp a Butterfly', 'DAMN']
album_positions = []
album_means = []

scatter_plots = []
fig, ax = plt.subplots(figsize=(10,5))

# Create scatter for each album
for idx, album in enumerate(album_list):
    # Calculate coordinates and plot
    y = rescale_df['gloom_index'][rescale_df.album == album].sort_values()
    x = (idx + 1) * np.ones((1, len(y)))
    scatter = ax.scatter(x, y, alpha=0.5, s=70)
    scatter_plots.append(scatter)
    
    # Store x-coordinate and mean gloom index of each album
    album_positions.append(idx + 1)
    album_means.append(y.mean())
    
    # Modify positions of labels to prevent overlapping
    modified_positions = []
    overlap_dist = 1.5
    for pos, val in enumerate(y):
        # Calculate modified position
        if pos != 0 and val - modified_positions[pos - 1] < overlap_dist:
            modified_positions.append((overlap_dist - (val - modified_positions[pos - 1])) + val)
            modified_positions[pos - 1] -= (overlap_dist - 1)
        else:
            modified_positions.append(val)
        ax.annotate(y.index[pos], (idx+1, y[pos]), xytext=(idx+1.1, modified_positions[pos]-0.6), size=14)
        
# Plot album averages
scatter_plots.append(ax.scatter(album_positions, album_means, c='black', s=80))
ax.plot(album_positions, album_means, c='black', linestyle='--')

# Plot styling
ax.set_axisbelow(True)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False  # labels along the bottom edge are off
)
ax.set_xticks(np.arange(1, len(album_list) + 1, 0.5))
ax.tick_params(
    axis='y', 
    which='both',
    labelsize='15'
)
ax.set_yticks(np.arange(0, 110, 10))
ax.legend(scatter_plots,
           album_list + ['Album Average'],
           scatterpoints=1,
           loc='upper center',
           bbox_to_anchor=(0.5, -0.05),
           ncol=6,
           fontsize=16
)
ax.set_title('Pulitzer Kenny\'s Song Gloominess by Album', fontsize=22)
ax.set_xlabel('Album', fontsize=17)
ax.set_ylabel('Gloom Index', fontsize=17)
ax.grid()
fig.set_size_inches(25, 17)

plt.savefig("result.png")
plt.show()

