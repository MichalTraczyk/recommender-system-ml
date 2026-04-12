from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_recommender_node
from ...modules.training_utils import prepare_dataloader


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=prepare_dataloader,
            inputs=["user_timelines_train", "parameters"],
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