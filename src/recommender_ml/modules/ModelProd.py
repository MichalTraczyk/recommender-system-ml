import math

import torch
import torch.nn as nn


class ProdMovieRecommender(nn.Module):
    def __init__(self, num_movies, num_genres, max_seq_len):
        super(ProdMovieRecommender, self).__init__()
        self.emb_dropout = nn.Dropout(0.2)
        self.d_model = 256
        self.movie_embedding = nn.Embedding(
            num_embeddings=num_movies,
            embedding_dim=256,
            padding_idx=0
        )
        self.input_norm = nn.LayerNorm(self.d_model)
        self.genre_embedding = nn.EmbeddingBag(
            num_embeddings=num_genres,
            embedding_dim=64,
            mode='mean',
            padding_idx=0
        )
        self.pos_embedding = nn.Embedding(max_seq_len, self.d_model)

        transformer_block = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=8, batch_first=True, dropout=0.2)

        self.transformer = nn.TransformerEncoder(transformer_block, num_layers=2)

        self.input_proj = nn.Linear(in_features=256 + 64, out_features=256)

        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1)
        )

    def forward(self, movie_ids, genre_ids):
        movies = self.movie_embedding(movie_ids)  #[B, seq_len, 256]

        batch_size, seq_len, max_genres = genre_ids.shape
        genre_ids_flat = genre_ids.view(batch_size * seq_len, max_genres)

        genres_flat = self.genre_embedding(genre_ids_flat)
        genres = genres_flat.view(batch_size, seq_len, -1)

        combined = torch.cat([movies, genres], dim=-1)
        combined = self.input_norm(self.input_proj(combined))
        combined = combined * math.sqrt(self.d_model)

        positions = torch.arange(seq_len, device=movie_ids.device)
        combined = combined + self.pos_embedding(positions)
        combined = self.emb_dropout(combined)

        padding_mask = (movie_ids == 0)
        mixed_sequence = self.transformer(combined, src_key_padding_mask=padding_mask)
        last_step = self.output_head(mixed_sequence[:, -1, :])
        return last_step
