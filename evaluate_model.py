import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import torch


def ndcg_score(relevance_scores, k):
    relevance_scores = relevance_scores[:k]
    dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance_scores))
    ideal_relevance = sorted(relevance_scores, reverse=True)[:k]
    ideal = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(ideal_relevance))

    return dcg / ideal if ideal != 0 else 0


# def predict(user_embedding, candidate_movie_embeddings):
#     user_embedding = user_embedding.reshape(1, -1)  # shape: (1, D)
#     scores = cosine_similarity(user_embedding, candidate_movie_embeddings)
#     return scores.flatten()  # shape: (num_candidates,)


def predict(model, user_ids, movie_ids, device="cpu"):
    """
    Given user and movie IDs, return similarity scores from the trained model.

    Args:
        model: Trained TwoTowerTripletNN model.
        user_ids: Tensor of user IDs.
        movie_ids: Tensor of movie IDs.
        device: 'cpu' or 'cuda'.

    Returns:
        scores: Tensor of similarity scores.
    """
    model.eval()
    model.to(device)

    # Move inputs to device
    user_ids = user_ids.to(device)
    movie_ids = movie_ids.to(device)

    # Get embeddings
    with torch.no_grad():
        user_emb = model.user_embedding(user_ids)
        user_vec = model.user_fc(user_emb)

        movie_emb = model.movie_embedding(movie_ids)
        movie_vec = model.movie_fc(movie_emb)

        # Similarity score: dot product
        scores = torch.sum(user_vec * movie_vec, dim=1)  # (N,)
        # OR: cosine similarity (optional)
        # scores = F.cosine_similarity(user_vec, movie_vec)

    return scores
