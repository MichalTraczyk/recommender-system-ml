import ast
import logging
import os
import numpy as mn
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def get_logger():
    return logging.getLogger(__name__)


def remap_ids(df):
    df['id'] = range(1, len(df) + 1)
    logger = get_logger()
    logger.info(f"\n--- Output of remap_ids ---\nShape: {df.shape}\n{df.head(3).to_string()}\n")
    return df


def extract_year(df):
    # extract year
    df['year'] = df['title'].str.extract(r'\((\d{4})\)', expand=False).astype('Int64')
    # remove extracted year from title
    df['title'] = df['title'].str.replace(r'\s*\(\d{4}\)', '', regex=True).str.strip()

    logger = get_logger()
    logger.info(f"\n--- Output of extract_year ---\nShape: {df.shape}\n{df.head(3).to_string()}\n")
    return df


# genre into hot encoding
def encode_all_genres(df, genre_col='genres'):
    genre_dummies = df[genre_col].str.get_dummies(sep='|')
    genre_dummies = genre_dummies.add_prefix('genre_')
    df = pd.concat([df, genre_dummies], axis=1)
    df.drop(genre_col, axis=1, inplace=True)
    # swap id column with title column, just for easier analysis later
    cols = list(df.columns)
    a, b = cols.index('title'), cols.index('id')
    cols[b], cols[a] = cols[a], cols[b]
    df = df[cols]

    genre_column_names = list(genre_dummies.columns)
    genre_df = pd.DataFrame({"genre_columns": genre_column_names})

    logger = get_logger()
    preview_cols = ['movieId', 'title', 'year', 'id'] + genre_column_names[:3]
    logger.info(f"\n--- Output of encode_all_genres ---\nShape: {df.shape}\n{df[preview_cols].head(3).to_string()}\n")
    return df, genre_df


def add_links(df: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    df = df.merge(links[['movieId', 'imdbId', 'tmdbId']], on='movieId', how='left')
    df['imdb_id'] = 'tt' + df['imdbId'].fillna(0).astype(int).astype(str).str.zfill(7)
    df['tmdb_id'] = df['tmdbId'].astype('Int64')
    df["poster_url"] = "ignore for now i will get it from the api"
    df = df.drop(columns=['imdbId', 'tmdbId'])
    return df


def add_poster_urls(df: pd.DataFrame):
    api_key = os.getenv("TMDB_API_KEY")
    tqdm.pandas(desc="Fetching Posters")
    df['poster_url'] = df['tmdb_id'].progress_apply(get_poster_url, api_key=api_key)
    return df


def get_poster_url(tmdb_id, api_key):
    if pd.isna(tmdb_id):
        return None

    BASE_IMG_URL = "https://image.tmdb.org/t/p/w500"
    url = f"https://api.themoviedb.org/3/movie/{int(tmdb_id)}?api_key={api_key}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            poster_path = data.get('poster_path')

            if poster_path:
                return BASE_IMG_URL + poster_path
    except Exception as e:
        print(f"Error fetching ID {tmdb_id}: {e}")

    return None

def compute_popularity_scores(
    movies: pd.DataFrame,
    ratings: pd.DataFrame,
    user_timelines: pd.DataFrame
) -> pd.DataFrame:
    # --- Rating-based score ---
    agg = ratings.groupby('movieId').agg(
        avg_rating=('rating', 'mean'),
        num_ratings=('rating', 'count')
    ).reset_index()
    agg['rating_score'] = agg['avg_rating'] * np.log1p(agg['num_ratings'])

    # --- Watch frequency score ---
    watch_counts = {}
    for seq in user_timelines['movie_sequence']:
        parsed = ast.literal_eval(seq) if isinstance(seq, str) else seq
        for movie_id in parsed:
            watch_counts[movie_id] = watch_counts.get(movie_id, 0) + 1

    watch_df = pd.DataFrame(
        watch_counts.items(), columns=['id', 'watch_count']  # 'id' = your remapped id
    )

    # --- Merge into movies ---
    df = movies.merge(agg[['movieId', 'avg_rating', 'num_ratings', 'rating_score']],
                      on='movieId', how='left')
    df = df.merge(watch_df, on='id', how='left')

    df['avg_rating'] = df['avg_rating'].fillna(0).round(2)
    df['num_ratings'] = df['num_ratings'].fillna(0).astype(int)
    df['rating_score'] = df['rating_score'].fillna(0).round(4)
    df['watch_count'] = df['watch_count'].fillna(0).astype(int)

    logger = get_logger()
    logger.info(f"Popularity scores added. Top 5:\n{df.nlargest(5, 'watch_count')[['title', 'avg_rating', 'num_ratings', 'watch_count']].to_string()}")
    return df