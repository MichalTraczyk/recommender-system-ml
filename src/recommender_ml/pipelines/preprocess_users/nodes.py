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