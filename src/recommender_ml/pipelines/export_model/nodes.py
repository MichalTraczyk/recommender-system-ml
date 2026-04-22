import logging
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import onnxruntime as ort
from tqdm import tqdm

from recommender_ml.modules.ModelExport import RecommenderFromEmbeddings
from recommender_ml.modules.training_utils import prepare_dataloader

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
    genre_weights = weights[1:len(genre_names) + 1]
    df['embedding'] = [list(row) for row in genre_weights]

    logger.info(f"Exported {len(df)} genre embeddings of dim {weights.shape[1]}")
    return df


def export_onnx_model(
        production_model: nn.Module,
        parameters: dict
) -> str:
    import onnx

    max_seq_len = parameters.get("max_sequence_length", 30)
    combined_dim = 256 + 64
    output_path = "data/06_models/production_model.onnx"

    wrapper = RecommenderFromEmbeddings(production_model, max_seq_len=max_seq_len)
    wrapper.eval()
    wrapper.to(torch.device("cpu"))

    dummy_input = torch.zeros(1, max_seq_len, combined_dim, dtype=torch.float32)

    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=["combined_embeddings"],
        output_names=["user_vector"],
        opset_version=14,
        dynamo=False,
        dynamic_axes={
            "combined_embeddings": {
                0: "batch_size"
            },
            "user_vector": {
                0: "batch_size"
            }
        }
    )
    logger.info(f"ONNX model exported to {output_path}")

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model verified")

    ort_session = ort.InferenceSession(output_path)

    test_input = torch.zeros(1, max_seq_len, combined_dim, dtype=torch.float32).numpy()
    out = ort_session.run(None, {"combined_embeddings": test_input})
    logger.info(f"Static seq_len={max_seq_len} verification passed → output shape: {out[0].shape}")

    return output_path


def evaluate_onnx_model(
        user_timelines_test: pd.DataFrame,
        parameters: dict,
        pytorch_model: torch.nn.Module
) -> pd.DataFrame:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_loader = prepare_dataloader(user_timelines_test, parameters)

    onnx_path = parameters.get("onnx_model_path", "data/06_models/production_model.onnx")
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if torch.cuda.is_available() else [
        'CPUExecutionProvider']

    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = session.get_inputs()[0].name

    pytorch_model.to(device)
    pytorch_model.eval()
    candidate_embeddings = pytorch_model.movie_embedding.weight[1:].detach()

    ranks_list = []

    with torch.no_grad():
        for m_seq, g_seq, target in tqdm(test_loader, desc="Evaluating ONNX Model"):
            m_seq = m_seq.to(device)
            g_seq = g_seq.to(device)
            target = target.to(device)

            movies = pytorch_model.movie_embedding(m_seq)
            batch_size, seq_len, max_genres = g_seq.shape
            genres_flat = pytorch_model.genre_embedding(g_seq.view(batch_size * seq_len, max_genres))
            genres = genres_flat.view(batch_size, seq_len, -1)

            combined = torch.cat([movies, genres], dim=-1)

            combined_np = combined.cpu().numpy()
            user_vectors_np = session.run(None, {input_name: combined_np})[0]

            user_vectors = torch.tensor(user_vectors_np, device=device)

            logits = torch.matmul(user_vectors, candidate_embeddings.T)

            target_indices = target - 1

            sorted_indices = torch.argsort(logits, dim=-1, descending=True)
            matches = (sorted_indices == target_indices.unsqueeze(1))

            ranks = matches.float().argmax(dim=1) + 1
            ranks_list.append(ranks.cpu())

    all_ranks = torch.cat(ranks_list).float()
    metrics = {"model": "production_onnx"}

    for k in [5, 10, 20]:
        hits = (all_ranks <= k).float()
        metrics[f'recall@{k}'] = round(hits.mean().item(), 4)

        ndcg = torch.where(all_ranks <= k, 1.0 / torch.log2(all_ranks + 1.0), torch.zeros_like(all_ranks))
        metrics[f'ndcg@{k}'] = round(ndcg.mean().item(), 4)

    metrics['mrr'] = round((1.0 / all_ranks).mean().item(), 4)

    cols = ['model', 'recall@5', 'ndcg@5', 'recall@10', 'ndcg@10', 'recall@20', 'ndcg@20', 'mrr']
    df_metrics = pd.DataFrame([metrics])[cols]

    return df_metrics