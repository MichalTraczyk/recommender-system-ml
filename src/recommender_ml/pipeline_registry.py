from kedro.pipeline import Pipeline
from .pipelines import preprocess_movies, preprocess_users, train


def register_pipelines() -> dict[str, Pipeline]:
    preprocess_movies_pipeline = preprocess_movies.create_pipeline()
    preprocess_users_pipeline = preprocess_users.create_pipeline()
    train_baseline_pipeline = train.create_pipeline()
    return {
        "preprocess_movies": preprocess_movies_pipeline,
        "preprocess_users": preprocess_users_pipeline,
        "train_baseline": train_baseline_pipeline
    }
