from kedro.pipeline import Pipeline
from .pipelines import preprocess_movies, preprocess_users


def register_pipelines() -> dict[str, Pipeline]:
    preprocess_movies_pipeline = preprocess_movies.create_pipeline()
    preprocess_users_pipeline = preprocess_users.create_pipeline()
    return {
        "preprocess_movies": preprocess_movies_pipeline,
        "preprocess_users": preprocess_users_pipeline,
    }
