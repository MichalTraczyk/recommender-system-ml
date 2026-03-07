from kedro.pipeline import Node, Pipeline
from .nodes import extract_year, remap_ids, encode_all_genres


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=remap_ids,
                inputs="raw_movies",
                outputs="movies_remapped",
                name="remap_ids_node"
            ),
            Node(
                func=extract_year,
                inputs="movies_remapped",
                outputs="movies_with_year",
                name="extract_year_node",
            ),
            Node(
                func=encode_all_genres,
                inputs="movies_with_year",
                outputs=["preprocessed_movies", "genre_column_list"],
                name="encode_all_genres_node"
            )
        ]
    )
