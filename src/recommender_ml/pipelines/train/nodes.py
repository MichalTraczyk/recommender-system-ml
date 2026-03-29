import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
import ast
import logging
from recommender_ml.modules.Model import BaselineMovieRecommender


def prepare_dataloader(user_timelines: pd.DataFrame, parameters: dict) -> DataLoader:
    logger = logging.getLogger(__name__)

    batch_size = parameters.get("batch_size", 32)
    max_seq_len = parameters.get("max_sequence_length", 10)
    max_genres = parameters.get("max_genres", 3)
    min_seq_len = parameters.get("min_sequence_length", 2)

    X_movies, X_genres, Y_targets = [], [], []

    logger.info("Creating dataloader")

    for _, row in user_timelines.iterrows():
        m_seq = ast.literal_eval(row['movie_sequence']) if isinstance(row['movie_sequence'], str) else row[
            'movie_sequence']
        g_seq = ast.literal_eval(row['genre_sequence']) if isinstance(row['genre_sequence'], str) else row[
            'genre_sequence']

        if len(m_seq) < 2:
            continue
        for target_idx in range(min_seq_len, len(m_seq)):
            target_movie = m_seq[target_idx]

            # All history before the target, capped at max_seq_len (most recent)
            context_movies = m_seq[:target_idx][-max_seq_len:]
            context_genres = g_seq[:target_idx][-max_seq_len:]

            pad_len = max_seq_len - len(context_movies)
            padded_movies = ([0] * pad_len) + context_movies
            padded_genres = ([[0] * max_genres] * pad_len) + context_genres

            X_movies.append(padded_movies)
            X_genres.append(padded_genres)
            Y_targets.append(target_movie)

    logger.info(f"Generated {len(X_movies)} training samples from {len(user_timelines)} users")

    dataset = TensorDataset(
        torch.tensor(X_movies, dtype=torch.long),
        torch.tensor(X_genres, dtype=torch.long),
        torch.tensor(Y_targets, dtype=torch.long)
    )

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    logger.info(f"Total Batches: {len(train_loader)}")

    return train_loader


def train_recommender_node(
        train_loader: torch.utils.data.DataLoader,
        num_movies: int,
        num_genres: int,
        parameters: dict
) -> nn.Module:
    logger = logging.getLogger(__name__)
    max_seq_len = parameters.get("max_sequence_length", 10)
    lr = parameters.get("learning_rate", 0.001)
    epochs = parameters.get("epochs", 5)

    model = BaselineMovieRecommender(num_movies=num_movies, num_genres=num_genres,max_seq_len=max_seq_len)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    model.train()
    logger.info(f"Starting training for {epochs} epochs.")

    for epoch in range(epochs):
        total_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{epochs}",
            unit="batch"
        )

        for movie_seq, genre_seq, target_movie in progress_bar:
            optimizer.zero_grad()
            user_vector = model(movie_seq, genre_seq)
            real_movie_embeddings = model.movie_embedding.weight[1:]
            logits = torch.matmul(user_vector, real_movie_embeddings.T)
            loss = criterion(logits, target_movie-1)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    logger.info("Training complete")

    return model