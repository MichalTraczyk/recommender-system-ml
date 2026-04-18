import math

import torch
import torch.nn as nn

from recommender_ml.modules.Model import BaselineMovieRecommender
from recommender_ml.modules.ModelProd import ProdMovieRecommender


class RecommenderFromEmbeddings(nn.Module):
    def __init__(self, base_model: ProdMovieRecommender):
        super().__init__()
        self.input_proj = base_model.input_proj
        self.input_norm = base_model.input_norm  # ← missing
        self.pos_embedding = base_model.pos_embedding
        self.transformer = base_model.transformer
        self.output_head = base_model.output_head  # ← missing
        self.d_model = base_model.d_model

    def forward(self, combined_embeddings: torch.Tensor) -> torch.Tensor:
        x = self.input_norm(self.input_proj(combined_embeddings))
        x = x * math.sqrt(self.d_model)

        seq_len = combined_embeddings.shape[1]
        positions = torch.arange(seq_len, device=combined_embeddings.device)
        x = x + self.pos_embedding(positions)

        padding_mask = (combined_embeddings.sum(dim=-1) == 0)
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return self.output_head(out[:, -1, :])