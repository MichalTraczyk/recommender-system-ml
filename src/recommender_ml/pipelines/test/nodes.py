import logging
import pandas as pd
import torch
from torch import nn
from recommender_ml.modules.training_utils import prepare_dataloader


def evaluate_model(
    model: nn.Module,
    user_timelines_test: pd.DataFrame,
    parameters: dict,
    model_name: str
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    k_values = [5, 10, 20]
    test_loader = prepare_dataloader(user_timelines_test, parameters)

    recalls = {k: 0 for k in k_values}
    ndcgs = {k: 0 for k in k_values}
    mrr_total = 0
    total = 0

    with torch.no_grad():
        for movie_seq, genre_seq, target in test_loader:
            movie_seq = movie_seq.to(device)
            genre_seq = genre_seq.to(device)
            target = target.to(device)

            user_vec = model(movie_seq, genre_seq)
            logits = torch.matmul(user_vec, model.movie_embedding.weight[1:].T)

            batch_size = target.size(0)
            total += batch_size

            # get rank of correct item for each sample in batch
            target_scores = logits.gather(1, (target - 1).unsqueeze(1))  # [batch, 1]
            ranks = (logits > target_scores).sum(dim=1) + 1

            # MRR
            mrr_total += (1.0 / ranks.float()).sum().item()

            for k in k_values:
                # Recall@K — was correct item in top K
                hit = (ranks <= k).float()
                recalls[k] += hit.sum().item()

                #discounted gain if item was in top K
                ndcg = (hit / torch.log2(ranks.float() + 1)).sum().item()
                ndcgs[k] += ndcg

    results = {"model": model_name}
    for k in k_values:
        results[f"recall@{k}"] = round(recalls[k] / total, 4)
        results[f"ndcg@{k}"] = round(ndcgs[k] / total, 4)
    results["mrr"] = round(mrr_total / total, 4)

    logger.info(f"\n--- {model_name} Evaluation Results ---")
    for key, val in results.items():
        if key != "model":
            logger.info(f"  {key}: {val}")

    return pd.DataFrame([results])


def evaluate_baseline_model(
    trained_recommender_model: nn.Module,
    user_timelines_test: pd.DataFrame,
    parameters: dict
) -> pd.DataFrame:
    return evaluate_model(trained_recommender_model, user_timelines_test, parameters, "baseline")


def evaluate_production_model(
    production_model: nn.Module,
    user_timelines_test: pd.DataFrame,
    parameters: dict
) -> pd.DataFrame:
    return evaluate_model(production_model, user_timelines_test, parameters, "production")