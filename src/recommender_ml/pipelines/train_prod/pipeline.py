from kedro.pipeline import Node, Pipeline
from .nodes import kfold_and_final_training


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=kfold_and_final_training,
            inputs=[
                "user_timelines_train",
                "parameters"
            ],
            outputs=["kfold_results", "production_model"],
            name="kfold_and_final_training_node",
        )
    ])