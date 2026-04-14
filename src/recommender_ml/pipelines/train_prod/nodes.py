import logging

import pandas as pd
import torch
from torch import nn
from recommender_ml.modules.training_utils import (
    run_kfold,
    train_final_model, prepare_dataloader,
)

logger = logging.getLogger(__name__)


def kfold_and_final_training(
        user_timelines_train: pd.DataFrame,
        parameters: dict
) -> tuple[pd.DataFrame, nn.Module]:
    num_movies = parameters.get("num_movies")
    num_genres = parameters.get("num_genres")

    results_df = run_kfold(
        user_timelines_train,
        num_movies,
        num_genres,
        parameters,
        prepare_dataloader
    )

    best_epoch = int(results_df["best_epoch"].mean().round())
    logger.info(f"Average best epoch across folds: {best_epoch} — using for final training")

    final_model = train_final_model(
        user_timelines_train,
        num_movies,
        num_genres,
        parameters,
        best_epoch,
        prepare_dataloader
    )

    return results_df, final_model
