import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from recommender_ml.modules.ModelExport import RecommenderFromEmbeddings

logger = logging.getLogger(__name__)


def export_movie_embeddings(
        production_model: nn.Module,
        preprocessed_movies: pd.DataFrame
) -> pd.DataFrame:
    df = preprocessed_movies[['id']].copy().reset_index(drop=True)
    movie_ids = torch.tensor(df['id'].values, dtype=torch.long)
    weights = production_model.movie_embedding.weight[movie_ids].detach().cpu().numpy()

    df['embedding'] = [list(row) for row in weights]

    logger.info(f"Exported {len(df)} movie embeddings of dim {weights.shape[1]}")
    return df


def export_genre_embeddings(
        production_model: nn.Module,
        genre_column_list: pd.DataFrame
) -> pd.DataFrame:
    weights = production_model.genre_embedding.weight.detach().cpu().numpy()
    genre_names = genre_column_list['genre_columns'].tolist()

    df = pd.DataFrame({'genre': genre_names})
    genre_weights = weights[1:len(genre_names) + 1]  # skip padding index 0
    df['embedding'] = [list(row) for row in genre_weights]

    logger.info(f"Exported {len(df)} genre embeddings of dim {weights.shape[1]}")
    return df


def export_onnx_model(
        production_model: nn.Module,
        parameters: dict
) -> str:
    import onnx
    import onnxruntime as ort

    max_seq_len = parameters.get("max_sequence_length", 10)
    combined_dim = 130  # 100 movie + 30 genre
    output_path = "data/06_models/production_model.onnx"

    wrapper = RecommenderFromEmbeddings(production_model)
    wrapper.eval()
    wrapper.to(torch.device("cpu"))

    dummy_input = torch.zeros(1, max_seq_len, combined_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["combined_embeddings"],
        output_names=["user_vector"],
        opset_version=17,
        dynamic_axes=None
    )
    logger.info(f"ONNX model exported to {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified")

    ort_session = ort.InferenceSession(output_path)
    out = ort_session.run(None, {"combined_embeddings": dummy_input.numpy()})
    logger.info(f"ONNX runtime check passed — output shape: {out[0].shape}")

    return output_path