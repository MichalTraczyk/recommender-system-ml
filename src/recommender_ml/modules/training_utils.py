import ast
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from recommender_ml.modules.Model import BaselineMovieRecommender


logger = logging.getLogger(__name__)

def bpr_loss(pos_scores: torch.Tensor, neg_scores: torch.Tensor) -> torch.Tensor:
    return -torch.log(torch.sigmoid(pos_scores - neg_scores) + 1e-8).mean()

def run_single_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        epoch_label: str = "Epoch"
) -> float:
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(loader, desc=epoch_label, unit="batch")

    for movie_seq, genre_seq, target_movie in progress_bar:
        movie_seq = movie_seq.to(device)
        genre_seq = genre_seq.to(device)
        target_movie = target_movie.to(device)

        optimizer.zero_grad()

        user_vector = model(movie_seq, genre_seq)
        logits = torch.matmul(user_vector, model.movie_embedding.weight[1:].T)

        pos_scores = logits.gather(1, (target_movie - 1).unsqueeze(1)).squeeze(1)
        neg_ids = torch.randint(0, logits.size(1), target_movie.shape, device=device)
        neg_scores = logits.gather(1, neg_ids.unsqueeze(1)).squeeze(1)

        loss = bpr_loss(pos_scores, neg_scores)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(loader)

def prepare_dataloader(user_timelines: pd.DataFrame, parameters: dict) -> DataLoader:
    logger = logging.getLogger(__name__)

    batch_size = parameters.get("batch_size", 256)
    max_seq_len = parameters.get("max_sequence_length", 10)
    max_genres = parameters.get("max_genres", 3)
    min_seq_len = parameters.get("min_sequence_length", 5)

    X_movies, X_genres, Y_targets = [], [], []

    logger.info(f"Creating dataloader batchsize: {batch_size}")

    for _, row in user_timelines.iterrows():
        m_seq = ast.literal_eval(row['movie_sequence']) if isinstance(row['movie_sequence'], str) else row[
            'movie_sequence']
        g_seq = ast.literal_eval(row['genre_sequence']) if isinstance(row['genre_sequence'], str) else row[
            'genre_sequence']

        if len(m_seq) < 2:
            continue
        for target_idx in range(min_seq_len, len(m_seq)):
            target_movie = m_seq[target_idx]

            context_movies = m_seq[:target_idx][-max_seq_len:]
            context_genres = g_seq[:target_idx][-max_seq_len:]

            pad_len = max_seq_len - len(context_movies)
            padded_movies = ([0] * pad_len) + context_movies
            padded_genres = [[0] * max_genres for _ in range(pad_len)] + context_genres

            X_movies.append(padded_movies)
            X_genres.append(padded_genres)
            Y_targets.append(target_movie)

    logger.info(f"Generated {len(X_movies)} training samples from {len(user_timelines)} users")

    dataset = TensorDataset(
        torch.tensor(X_movies, dtype=torch.long),
        torch.tensor(X_genres, dtype=torch.long),
        torch.tensor(Y_targets, dtype=torch.long)
    )

    train_loader = DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)
    logger.info(f"Total Batches: {len(train_loader)}")

    return train_loader


def run_validation_epoch(
        model: nn.Module,
        loader: DataLoader,
        device: torch.device
) -> float:
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for movie_seq, genre_seq, target_movie in loader:
            movie_seq = movie_seq.to(device)
            genre_seq = genre_seq.to(device)
            target_movie = target_movie.to(device)

            user_vector = model(movie_seq, genre_seq)
            logits = torch.matmul(user_vector, model.movie_embedding.weight[1:].T)

            pos_scores = logits.gather(1, (target_movie - 1).unsqueeze(1)).squeeze(1)
            neg_ids = torch.randint(0, logits.size(1), target_movie.shape, device=device)
            neg_scores = logits.gather(1, neg_ids.unsqueeze(1)).squeeze(1)

            total_loss += bpr_loss(pos_scores, neg_scores).item()

    return total_loss / len(loader)


def recall_at_k(
        model: nn.Module,
        loader: DataLoader,
        k: int = 10,
        device: torch.device = torch.device("cpu")
) -> float:
    model.eval()
    hits, total = 0, 0

    with torch.no_grad():
        for movie_seq, genre_seq, target in loader:
            movie_seq = movie_seq.to(device)
            genre_seq = genre_seq.to(device)
            target = target.to(device)

            user_vec = model(movie_seq, genre_seq)
            logits = torch.matmul(user_vec, model.movie_embedding.weight[1:].T)
            topk = logits.topk(k, dim=1).indices + 1
            hits += (topk == target.unsqueeze(1)).any(dim=1).sum().item()
            total += target.size(0)

    model.train()
    return hits / total

def build_model(
        num_movies: int,
        num_genres: int,
        max_seq_len: int,
        device: torch.device
) -> nn.Module:
    model = BaselineMovieRecommender(
        num_movies=num_movies,
        num_genres=num_genres,
        max_seq_len=max_seq_len
    )
    model.to(device)
    return model


def build_optimizer(
        model: nn.Module,
        parameters: dict
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    lr = parameters.get("learning_rate", 3e-4)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=1
    )
    return optimizer, scheduler

def train_with_early_stopping(
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        num_movies: int,
        num_genres: int,
        parameters: dict,
        prepare_dataloader_fn
) -> tuple[nn.Module, int, float]:
    max_epochs = parameters.get("max_epochs", 20)
    patience = parameters.get("patience", 3)
    max_sequence_len = parameters.get("max_sequence_length", 10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader = prepare_dataloader_fn(train_df, parameters)
    val_loader = prepare_dataloader_fn(val_df, parameters)

    model = build_model(num_movies, num_genres, max_sequence_len, device)
    optimizer, scheduler = build_optimizer(model, parameters)

    best_val_loss = float('inf')
    best_epoch = 0
    best_weights = None
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        avg_train_loss = run_single_epoch(
            model, train_loader, optimizer, device,
            epoch_label=f"Epoch {epoch + 1} [train]"
        )
        avg_val_loss = run_validation_epoch(model, val_loader, device)
        scheduler.step(avg_val_loss)

        logger.info(
            f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | Best Epoch: {best_epoch + 1}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
            logger.info(f"New best model at epoch {epoch + 1}")
        else:
            epochs_no_improve += 1
            logger.info(f"No improvement {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}, best was {best_epoch + 1}")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    return model, best_epoch + 1, best_val_loss


def run_kfold(
        user_timelines: pd.DataFrame,
        num_movies: int,
        num_genres: int,
        parameters: dict,
        prepare_dataloader_fn
) -> pd.DataFrame:
    k = parameters.get("n_folds", 5)
    lr = parameters.get("learning_rate", 3e-4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kf = KFold(n_splits=k, shuffle=True, random_state=parameters.get("random_seed", 42))

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(user_timelines)):
        logger.info(f"─── Fold {fold + 1}/{k} ───")

        fold_train_df = user_timelines.iloc[train_idx].reset_index(drop=True)
        fold_val_df = user_timelines.iloc[val_idx].reset_index(drop=True)

        model, best_epoch, best_val_loss = train_with_early_stopping(
            fold_train_df, fold_val_df,
            num_movies, num_genres,
            parameters, prepare_dataloader_fn
        )

        val_loader = prepare_dataloader_fn(fold_val_df, parameters)
        recall = recall_at_k(model, val_loader, k=10, device=device)

        fold_scores.append({
            "fold": fold + 1,
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 4),
            "recall@10": round(recall, 4),
            "learning_rate": lr
        })

        logger.info(
            f"Fold {fold + 1} done | Best Epoch: {best_epoch} | "
            f"Val Loss: {best_val_loss:.4f} | Recall@10: {recall:.4f}"
        )

    results_df = pd.DataFrame(fold_scores)
    logger.info(
        f"K-Fold complete | Mean Recall@10: {results_df['recall@10'].mean():.4f} | "
        f"Mean Val Loss: {results_df['best_val_loss'].mean():.4f} | "
        f"Mean Best Epoch: {results_df['best_epoch'].mean():.1f}"
    )
    return results_df


def train_final_model(
        user_timelines: pd.DataFrame,
        num_movies: int,
        num_genres: int,
        parameters: dict,
        num_epochs: int,
        prepare_dataloader_fn
) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Starting training on device: {device}")
    train_loader = prepare_dataloader_fn(user_timelines, parameters)
    max_sequence_len = parameters.get("max_sequence_length", 10)
    model = build_model(num_movies, num_genres, max_sequence_len, device)

    optimizer, _ = build_optimizer(model, parameters)

    for epoch in range(num_epochs):
        avg_loss = run_single_epoch(
            model, train_loader, optimizer, device,
            epoch_label=f"Final Epoch {epoch + 1}/{num_epochs}"
        )
        logger.info(f"Final Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.4f}")

    logger.info("Production model training complete")
    return model