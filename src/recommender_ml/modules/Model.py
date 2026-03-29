import torch
import torch.nn as nn


class BaselineMovieRecommender(nn.Module):
    def __init__(self, num_movies, num_genres,max_seq_len):
        super().__init__()
        self.d_model = 130
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


        transformer_block = nn.TransformerEncoderLayer(d_model=130, nhead=10, batch_first=True)

        self.transformer = nn.TransformerEncoder(transformer_block, num_layers=3)

        self.squish_layer = nn.Linear(in_features=130, out_features=100)

    def forward(self, movie_ids, genre_ids):
        movies = self.movie_embedding(movie_ids)  # Shape: [x, 100]

        batch_size, seq_len, max_genres = genre_ids.shape
        genre_ids_flat = genre_ids.view(batch_size * seq_len, max_genres)

        genres_flat = self.genre_embedding(genre_ids_flat)
        genres = genres_flat.view(batch_size, seq_len, -1)
        combined = torch.cat([movies, genres], dim=-1)

        positions = torch.arange(seq_len, device=movie_ids.device)
        combined = combined + self.pos_embedding(positions)

        padding_mask = (movie_ids == 0)
        mixed_sequence = self.transformer(combined, src_key_padding_mask=padding_mask)
        last_step = mixed_sequence[:, -1, :]
        final_vector = self.squish_layer(last_step)
        return final_vector