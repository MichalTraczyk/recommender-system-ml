import logging

import pandas as pd
from sympy.printing.pytorch import torch
from torch import nn

from recommender_ml.pipelines.train_baseline.nodes import prepare_dataloader, train_recommender_node, bpr_loss


def recall_at_k(model, loader, k=10, device='cpu'):
    model.eval()
    hits, total = 0, 0
    with torch.no_grad():
        for movie_seq, genre_seq, target in loader:
            movie_seq, genre_seq, target = movie_seq.to(device), genre_seq.to(device), target.to(device)
            user_vec = model(movie_seq, genre_seq)
            logits = torch.matmul(user_vec, model.movie_embedding.weight[1:].T)
            topk = logits.topk(k, dim=1).indices + 1
            hits += (topk == target.unsqueeze(1)).any(dim=1).sum().item()
            total += target.size(0)
    model.train()
    return hits / total


def kfold_and_final_training(
    user_timelines_train: pd.DataFrame,
    parameters: dict
) -> tuple[pd.DataFrame, nn.Module]:
    import torch
    from sklearn.model_selection import KFold

    logger = logging.getLogger(__name__)

    k = parameters.get("n_folds", 5)
    lr = parameters.get("learning_rate", 3e-4)
    num_movies = parameters.get("num_movies")
    num_genres = parameters.get("num_genres")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=parameters.get("random_seed", 42))

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(user_timelines_train)):
        logger.info(f"Starting fold {fold + 1}/{k}")

        fold_train_df = user_timelines_train.iloc[train_idx].reset_index(drop=True)
        fold_val_df = user_timelines_train.iloc[val_idx].reset_index(drop=True)

        train_loader = prepare_dataloader(fold_train_df, parameters)
        val_loader = prepare_dataloader(fold_val_df, parameters)

        model = train_recommender_node(train_loader, num_movies, num_genres, parameters)
        final_lr = lr
        avg_loss = sum(
            bpr_loss(
                torch.matmul(model(ms.to(device), gs.to(device)), model.movie_embedding.weight[1:].T)
                .gather(1, (t.to(device) - 1).unsqueeze(1)).squeeze(1),
                torch.matmul(model(ms.to(device), gs.to(device)), model.movie_embedding.weight[1:].T)
                .gather(1, torch.randint(0, num_movies - 1, t.shape).to(device).unsqueeze(1)).squeeze(1)
            ).item()
            for ms, gs, t in val_loader
        ) / len(val_loader)

        recall = recall_at_k(model, val_loader, k=10, device=str(device))

        fold_scores.append({
            "fold": fold + 1,
            "recall@10": round(recall, 4),
            "avg_val_loss": round(avg_loss, 4),
            "learning_rate": final_lr
        })
        logger.info(f"Fold {fold + 1} | Recall@10: {recall:.4f} | Val Loss: {avg_loss:.4f}")

    results_df = pd.DataFrame(fold_scores)
    mean_recall = results_df["recall@10"].mean()
    mean_loss = results_df["avg_val_loss"].mean()
    logger.info(f"K-Fold complete | Mean Recall@10: {mean_recall:.4f} | Mean Val Loss: {mean_loss:.4f}")

    logger.info("Starting final production model training on full train split...")
    full_train_loader = prepare_dataloader(user_timelines_train, parameters)
    final_model = train_recommender_node(full_train_loader, num_movies, num_genres, parameters)

    return results_df, final_model