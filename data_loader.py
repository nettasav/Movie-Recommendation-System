import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

# Define a threshold to classify movies as "liked" or "not liked"
POSITIVE_THRESHOLD = 4.0  # Ratings >= 4 are considered "positive"


class MovieLensTripletDataset(Dataset):
    def __init__(self, df):
        
        triplets = self.generate_triplets(df)
        self.users = torch.tensor([t[0] for t in triplets], dtype=torch.long)
        self.pos_movies = torch.tensor([t[1] for t in triplets], dtype=torch.long)
        self.neg_movies = torch.tensor([t[2] for t in triplets], dtype=torch.long)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, idx):
        return self.users[idx], self.pos_movies[idx], self.neg_movies[idx]


    # Create triplets: (user, positive_movie, negative_movie)
    def generate_triplets(self, df):
        triplets = []
        grouped = df.groupby('userId')

        for user, user_data in grouped:
            pos_movies = user_data[user_data['rating'] >= POSITIVE_THRESHOLD]['movieId'].values
            neg_movies = user_data[user_data['rating'] < POSITIVE_THRESHOLD]['movieId'].values

            if len(pos_movies) == 0 or len(neg_movies) == 0:
                continue  # Skip users without positive and negative ratings

            for pos_movie in pos_movies:
                neg_movie = np.random.choice(neg_movies)  # Randomly sample a negative movie
                triplets.append((user, pos_movie, neg_movie))

        return triplets