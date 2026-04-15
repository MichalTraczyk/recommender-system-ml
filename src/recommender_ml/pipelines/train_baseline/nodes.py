import logging
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
import torch.optim as optim
import ast

from recommender_ml.modules.training_utils import (
    build_model,
    build_optimizer,
    run_single_epoch,
)

logger = logging.getLogger(__name__)

def train_recommender_node(
        train_loader: DataLoader,
        num_movies: int,
        num_genres: int,
        parameters: dict
) -> nn.Module:
    epochs = parameters.get("epochs", 5)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info(f"Starting baseline training on: {device}")

    model = build_model(num_movies, num_genres, parameters.get("max_sequence_length"), device, "baseline")
    optimizer, scheduler = build_optimizer(model, parameters)

    for epoch in range(epochs):
        avg_loss = run_single_epoch(
            model, train_loader, optimizer, device,
            epoch_label=f"Epoch {epoch + 1}/{epochs}"
        )
        scheduler.step(avg_loss)
        logger.info(f"Epoch {epoch + 1}/{epochs} | Average Loss: {avg_loss:.4f}")

    logger.info("Baseline training complete")
    return model