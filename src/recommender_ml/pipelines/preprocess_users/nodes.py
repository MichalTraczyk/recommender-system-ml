import ast
import logging
import pandas as pd

def get_logger():
    return logging.getLogger(__name__)


def merge_movie_ids(ratings: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    mapping_df = movies[['movieId', 'id']]

    merged_df = ratings.merge(mapping_df, on='movieId', how='inner')

    logger = get_logger()
    logger.info(f"\n--- Output of merge_movie_ids ---\nShape: {merged_df.shape}\n{merged_df[['userId', 'movieId', 'id', 'timestamp']].head(3).to_string()}\n")

    return merged_df


def build_user_timelines(df: pd.DataFrame, min_history: int = 10) -> pd.DataFrame:
    df = df.sort_values(by=['userId', 'timestamp'])
    timelines = df.groupby('userId')['id'].apply(list).reset_index(name='movie_sequence')
    timelines = timelines[timelines['movie_sequence'].map(len) >= min_history]

    logger = get_logger()
    logger.info(
        f"\n--- Output of build_user_timelines ---\nUsers Retained: {len(timelines)}\n{timelines.head(3).to_string()}\n")

    return timelines


def enrich_user_timelines(
        user_timelines: pd.DataFrame,
        preprocessed_movies: pd.DataFrame,
        parameters: dict
) -> pd.DataFrame:
    max_genres = parameters.get("max_genres", 3)
    genre_cols = [col for col in preprocessed_movies.columns if col.startswith('genre_')]
    genre_to_idx = {col: idx + 1 for idx, col in enumerate(genre_cols)}

    movie_to_genres = {}
    for _, row in preprocessed_movies.iterrows():
        m_id = row['id']
        active_genres = [genre_to_idx[col] for col in genre_cols if row[col] == 1]

        padded_genres = active_genres[:max_genres] + [0] * max(0, max_genres - len(active_genres))
        movie_to_genres[m_id] = padded_genres

    def build_genre_seq(movie_seq):
        if isinstance(movie_seq, str):
            movie_seq = ast.literal_eval(movie_seq)

        return [movie_to_genres.get(m_id, [0] * max_genres) for m_id in movie_seq]

    user_timelines['genre_sequence'] = user_timelines['movie_sequence'].apply(build_genre_seq)
    user_timelines['genre_sequence'] = user_timelines['genre_sequence'].astype(str)

    logger = get_logger()
    logger.info(
        f"\n--- Output of enrich_user_timelines ---\nShape: {user_timelines.shape}\n{user_timelines[['userId', 'movie_sequence', 'genre_sequence']].head(3).to_string()}\n")

    return user_timelines


def split_user_timelines(timelines: pd.DataFrame, parameters: dict):
    test_size = parameters.get("test_size", 0.2)
    seed = parameters.get("random_seed", 42)

    train_idx = timelines.sample(frac=1 - test_size, random_state=seed).index
    train = timelines.loc[train_idx].reset_index(drop=True)
    test = timelines.drop(train_idx).reset_index(drop=True)

    logger = get_logger()
    logger.info(f"Train users: {len(train)}, Test users: {len(test)}")

    return train, test