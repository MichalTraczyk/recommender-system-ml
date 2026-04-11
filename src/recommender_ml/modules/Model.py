import math

import torch
import torch.nn as nn


class BaselineMovieRecommender(nn.Module):
    def __init__(self, num_movies, num_genres,max_seq_len):
        super(BaselineMovieRecommender,self).__init__()
        self.emb_dropout = nn.Dropout(0.2)
        self.d_model = 100
        self.movie_embedding = nn.Embedding(
            num_embeddings=num_movies,
            embedding_dim=100,
            padding_idx=0
        )

        self.genre_embedding = nn.EmbeddingBag(
            num_embeddings=num_genres,
            embedding_dim=30,
            mode='mean',
            padding_idx=0
        )
        self.pos_embedding = nn.Embedding(max_seq_len, self.d_model)


        transformer_block = nn.TransformerEncoderLayer(d_model=self.d_model, nhead=4, batch_first=True)

        self.transformer = nn.TransformerEncoder(transformer_block, num_layers=1)

        self.input_proj = nn.Linear(in_features=130, out_features=100)

    def forward(self, movie_ids, genre_ids):
        movies = self.movie_embedding(movie_ids)  # Shape: [x, 100]

        batch_size, seq_len, max_genres = genre_ids.shape
        genre_ids_flat = genre_ids.view(batch_size * seq_len, max_genres)

        genres_flat = self.genre_embedding(genre_ids_flat)
        genres = genres_flat.view(batch_size, seq_len, -1)

        combined = torch.cat([movies, genres], dim=-1)
        combined = self.input_proj(combined)
        combined = combined * math.sqrt(self.d_model)


        positions = torch.arange(seq_len, device=movie_ids.device)
        combined = combined + self.pos_embedding(positions)
        combined = self.emb_dropout(combined)

        padding_mask = (movie_ids == 0)
        mixed_sequence = self.transformer(combined, src_key_padding_mask=padding_mask)
        last_step = mixed_sequence[:, -1, :]
        return last_step