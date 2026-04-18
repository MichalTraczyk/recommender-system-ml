import math

import torch
import torch.nn as nn

from recommender_ml.modules.Model import BaselineMovieRecommender
from recommender_ml.modules.ModelProd import ProdMovieRecommender


class RecommenderFromEmbeddings(nn.Module):
    def __init__(self, base_model: ProdMovieRecommender):
        super().__init__()
        self.input_proj = base_model.input_proj
        self.input_norm = base_model.input_norm
        self.transformer = base_model.transformer
        self.output_head = base_model.output_head
        self.d_model = base_model.d_model

    def _sinusoidal_pe(self, seq_len: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, combined_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_proj(combined_embeddings))
        x = x * math.sqrt(self.d_model)
        x = x + self._sinusoidal_pe(combined_embeddings.shape[1], combined_embeddings.device)

        padding_mask = (combined_embeddings.sum(dim=-1) == 0)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.output_head(out[:, -1, :])