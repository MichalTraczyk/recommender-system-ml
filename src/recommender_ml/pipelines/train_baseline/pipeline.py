from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_recommender_node, prepare_dataloader


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_dataloader,
            inputs=["user_timelines_train", "params:model_training"],
            outputs="train_dataloader",
            name="prepare_dataloader_node"
        ),
        node(
            func=train_recommender_node,
            inputs=[
                "train_dataloader",
                "params:num_movies",
                "params:num_genres",
                "params:model_training"
            ],
            outputs="trained_recommender_model",
            name="train_recommender_model_node"
        )
    ])