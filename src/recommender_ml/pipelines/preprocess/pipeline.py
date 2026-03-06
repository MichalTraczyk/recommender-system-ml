from kedro.pipeline import Node, Pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                func="",
                inputs="shuttles_excel",
                outputs="shuttles@csv",
                name="load_shuttles_to_csv_node",
            )
        ]
    )
