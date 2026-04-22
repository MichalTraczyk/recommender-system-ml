from kedro.pipeline import Node, Pipeline
from .nodes import export_movie_embeddings, export_genre_embeddings, export_onnx_model, evaluate_onnx_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=export_movie_embeddings,
            inputs=["production_model", "preprocessed_movies"],
            outputs="movie_embeddings_csv",
            name="export_movie_embeddings_node",
        ),
        Node(
            func=export_genre_embeddings,
            inputs=["production_model", "genre_column_list"],
            outputs="genre_embeddings_csv",
            name="export_genre_embeddings_node",
        ),
        Node(
            func=export_onnx_model,
            inputs=["production_model", "parameters"],
            outputs="onnx_model_path",
            name="export_onnx_model_node",
        ),
        Node(
            func=evaluate_onnx_model,
            inputs=["user_timelines_test", "parameters", "production_model"],
            outputs="onnx_metrics",
            name="evaluate_onnx_model_node"

        )
    ])
