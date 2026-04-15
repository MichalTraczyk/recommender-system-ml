import math

import torch
import torch.nn as nn

from recommender_ml.modules.Model import BaselineMovieRecommender


class RecommenderFromEmbeddings(nn.Module):
    def __init__(self, base_model: BaselineMovieRecommender):
        super().__init__()
        self.input_proj = base_model.input_proj
        self.pos_embedding = base_model.pos_embedding
        self.transformer = base_model.transformer
        self.d_model = base_model.d_model

    def forward(self, combined_embeddings: torch.Tensor) -> torch.Tensor:
        # combined_embeddings: [1, seq_len, 130] (movie_emb 100 + genre_emb 30)
        x = self.input_proj(combined_embeddings)
        x = x * math.sqrt(self.d_model)

        seq_len = combined_embeddings.shape[1]
        positions = torch.arange(seq_len, device=combined_embeddings.device)
        x = x + self.pos_embedding(positions)

        padding_mask = (combined_embeddings.sum(dim=-1) == 0)  # detect padding
        out = self.transformer(x, src_key_padding_mask=padding_mask)
        return out[:, -1, :]  # [1, 100] user vector