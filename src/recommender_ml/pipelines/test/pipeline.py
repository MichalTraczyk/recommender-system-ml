from kedro.pipeline import Node, Pipeline
from .nodes import evaluate_baseline_model, evaluate_production_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=evaluate_baseline_model,
            inputs=[
                "trained_recommender_model",
                "user_timelines_test",
                "parameters"
            ],
            outputs="baseline_evaluation_results",
            name="evaluate_baseline_model_node",
        ),
        Node(
            func=evaluate_production_model,
            inputs=[
                "production_model",
                "user_timelines_test",
                "parameters"
            ],
            outputs="production_evaluation_results",
            name="evaluate_production_model_node",
        ),
    ])