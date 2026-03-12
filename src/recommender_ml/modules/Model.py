import torch
import torch.nn as nn
class TrainableMovieRecommender(nn.Module):
    def __init__(self):
        super().__init__()
        self.movie_embedding = nn.Embedding(num_embeddings=10000, embedding_dim=100)
        self.genre_embedding = nn.Embedding(num_embeddings=20, embedding_dim=30)

        transformer_block = nn.TransformerEncoderLayer(d_model=130, nhead=10, batch_first=True)

        self.transformer = nn.TransformerEncoder(transformer_block, num_layers=3)

        self.squish_layer = nn.Linear(in_features=130, out_features=100)

    def forward(self, movie_ids, genre_ids):
        movies = self.movie_embedding(movie_ids)  # Shape: [x, 100]
        genres = self.genre_embedding(genre_ids)  # Shape: [x, 30]

        combined = torch.cat([movies, genres], dim=-1)

        mixed_sequence = self.transformer(combined)
        last_step = mixed_sequence[:, -1, :]
        final_vector = self.squish_layer(last_step)
        return final_vector