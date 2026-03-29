import logging

import pandas as pd

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
    #swap id column with title column, just for easier analysis later
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
