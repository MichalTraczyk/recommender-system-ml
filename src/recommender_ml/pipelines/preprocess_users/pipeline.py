from kedro.pipeline import Node, Pipeline

from .nodes import merge_movie_ids, build_user_timelines, enrich_user_timelines


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func=merge_movie_ids,
                inputs=["raw_ratings", "preprocessed_movies"],
                outputs="ratings_with_mapped_ids",
                name="merge_movie_ids_node",
            ),
            Node(
                func=build_user_timelines,
                inputs="ratings_with_mapped_ids",
                outputs="user_movie_timeline",
                name="build_user_timelines_node",
            ),
            Node(
                func=enrich_user_timelines,
                inputs=["user_movie_timeline","preprocessed_movies","parameters"],
                outputs="user_timelines",
                name="enrich_user_timelines_node"
            )
        ]
    )