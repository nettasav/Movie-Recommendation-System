import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle


class TwoTowerTripletNN(nn.Module):
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super(TwoTowerTripletNN, self).__init__()

        # User Tower
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.user_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

        # Movie Tower
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        self.movie_fc = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32)
        )

    def forward(self, user_ids, pos_movie_ids, neg_movie_ids):
        # Encode user
        user_emb = self.user_embedding(user_ids)
        user_vec = self.user_fc(user_emb)

        # Encode positive movie
        pos_emb = self.movie_embedding(pos_movie_ids)
        pos_vec = self.movie_fc(pos_emb)

        # Encode negative movie
        neg_emb = self.movie_embedding(neg_movie_ids)
        neg_vec = self.movie_fc(neg_emb)

        return user_vec, pos_vec, neg_vec

    def fit(self, train_loader, val_loader, criterion, optimizer, epochs=10, device='cpu'):
        self.to(device)
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            self.train()
            total_train_loss = 0

            for users, pos_movies, neg_movies in train_loader:
                users, pos_movies, neg_movies = users.to(device), pos_movies.to(device), neg_movies.to(device)

                optimizer.zero_grad()
                user_vec, pos_vec, neg_vec = self(users, pos_movies, neg_movies)
                loss = criterion(user_vec, pos_vec, neg_vec)
                loss.backward()
                optimizer.step()

                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation
            self.eval()
            total_val_loss = 0
            with torch.no_grad():
                for users, pos_movies, neg_movies in val_loader:
                    users, pos_movies, neg_movies = users.to(device), pos_movies.to(device), neg_movies.to(device)
                    user_vec, pos_vec, neg_vec = self(users, pos_movies, neg_movies)
                    loss = criterion(user_vec, pos_vec, neg_vec)
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        return train_losses, val_losses

    def save_model(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load_model(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, user_vec, pos_vec, neg_vec):
        pos_distance = F.pairwise_distance(user_vec, pos_vec, p=2)  # Euclidean distance
        neg_distance = F.pairwise_distance(user_vec, neg_vec, p=2)
        loss = torch.clamp(pos_distance - neg_distance + self.margin, min=0.0).mean()
        return loss
