import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import numpy as np

from triplet_model import TwoTowerTripletNN, TripletLoss
from data_loader import MovieLensTripletDataset, generate_triplets

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def train_model(train_df, val_df):
    # user_encoder = LabelEncoder()
    # movie_encoder = LabelEncoder()
    # df["userId"] = user_encoder.fit_transform(df["userId"])
    # df["movieId"] = movie_encoder.fit_transform(df["movieId"])

    # num_users = df["userId"].nunique()
    # num_movies = df["movieId"].nunique()

    # unique_users = df["userId"].unique()
    # train_users, test_users = train_test_split(
    #     unique_users, test_size=0.2, random_state=42
    # )
    # train_users, val_users = train_test_split(
    #     train_users, test_size=0.1, random_state=42
    # )

    # train_df = df[df["userId"].isin(train_users)]
    # val_df = df[df["userId"].isin(val_users)]
    # test_df = df[df["userId"].isin(test_users)]

    # train_triplets = generate_triplets(train_df)
    # val_triplets = generate_triplets(val_df)
    # test_triplets = generate_triplets(test_df)

    # Extract number of users and movies from the data
    num_users = max(train_df["userId"].max(), val_df["userId"].max()) + 1
    num_movies = max(train_df["movieId"].max(), val_df["movieId"].max()) + 1

    train_dataset = MovieLensTripletDataset(train_df)
    val_dataset = MovieLensTripletDataset(val_df)
    # test_dataset = MovieLensTripletDataset(test_df)

    BATCH_SIZE = 256

    train_loader = DataLoader(
        train_dataset.generate_triplets(), batch_size=BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset.generate_triplets(), batch_size=BATCH_SIZE, shuffle=False
    )
    # test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    EPOCHS = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TwoTowerTripletNN(num_users, num_movies, embedding_dim=64).to(device)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    # Store losses for visualization
    train_losses = []
    val_losses = []

    # Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0

        for users, pos_movies, neg_movies in train_loader:
            users, pos_movies, neg_movies = (
                users.to(device),
                pos_movies.to(device),
                neg_movies.to(device),
            )

            optimizer.zero_grad()
            user_vec, pos_vec, neg_vec = model(users, pos_movies, neg_movies)
            loss = criterion(user_vec, pos_vec, neg_vec)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation step
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for users, pos_movies, neg_movies in val_loader:
                users, pos_movies, neg_movies = (
                    users.to(device),
                    pos_movies.to(device),
                    neg_movies.to(device),
                )
                user_vec, pos_vec, neg_vec = model(users, pos_movies, neg_movies)
                loss = criterion(user_vec, pos_vec, neg_vec)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
        )

    # return train_losses, val_losses
    return model
