import math

import torch
import torch.nn as nn


class ProdMovieRecommender(nn.Module):
    def __init__(self, num_movies, num_genres):
        super(ProdMovieRecommender, self).__init__()
        self.emb_dropout = nn.Dropout(0.2)
        self.d_model = 256
        self.movie_embedding = nn.Embedding(num_embeddings=num_movies, embedding_dim=256, padding_idx=0)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.genre_embedding = nn.EmbeddingBag(num_embeddings=num_genres, embedding_dim=64, mode='mean', padding_idx=0)
        transformer_block = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True, dropout=0.2)
        self.transformer = nn.TransformerEncoder(transformer_block, num_layers=2)
        self.input_proj = nn.Linear(in_features=256 + 64, out_features=256)
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def _sinusoidal_pe(self, seq_len: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(seq_len, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, device=device) * (-math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(seq_len, self.d_model, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def forward(self, movie_ids, genre_ids):
        movies = self.movie_embedding(movie_ids)

        batch_size, seq_len, max_genres = genre_ids.shape
        genres_flat = self.genre_embedding(genre_ids.view(batch_size * seq_len, max_genres))
        genres = genres_flat.view(batch_size, seq_len, -1)

        combined = torch.cat([movies, genres], dim=-1)
        combined = self.input_norm(self.input_proj(combined))
        combined = combined * math.sqrt(self.d_model)
        combined = combined + self._sinusoidal_pe(seq_len, movie_ids.device)
        combined = self.emb_dropout(combined)

        padding_mask = (movie_ids == 0)
        mixed_sequence = self.transformer(combined, src_key_padding_mask=padding_mask)
        return self.output_head(mixed_sequence[:, -1, :])
