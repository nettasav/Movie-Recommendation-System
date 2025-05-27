from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from random import randint
import pandas as pd
from datetime import datetime, timedelta, timezone

from sklearn.metrics import ndcg_score

from sqlalchemy import create_engine
import psycopg2
import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from triplet_model import TwoTowerTripletNN, TripletLoss
from data_loader import MovieLensTripletDataset, generate_triplets
from train import train_model
from preprocess import split_data
import pickle
from evaluate_model import predict, ndcg_score


def _data_ingestion():
    # dummy task
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    with engine.connect() as con:
        rs = con.execute("SELECT * FROM data LIMIT 10")

    for row in rs:
        print(row)


def _create_movie_ranking_table():
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")

    query = """
            CREATE OR REPLACE VIEW ratings_view
            AS 
            WITH ratings AS (
            SELECT movie_id, AVG(rating) as avg_rating  
            FROM data
            GROUP BY movie_id
            )

            SELECT ratings.*, item.movie_title
            FROM ratings
            LEFT JOIN item ON item.movie_id=ratings.movie_id
            ORDER BY avg_rating DESC;
            """
    query2 = """ SELECT * FROM ratings_view LIMIT 10;"""

    with engine.connect() as con:
        con.execute(query)
        rs = con.execute(query2)  # Create logs

    for row in rs:
        print(row)


def _create_watching_list_table():
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = """
            CREATE OR REPLACE VIEW watching_list AS 
            SELECT user_id, array_agg(movie_id) as watched_movies , count(movie_id) as num_watched
            FROM  data
            GROUP BY user_id
            """
    query2 = """ SELECT * FROM watching_list ORDER BY user_id LIMIT 10; """

    with engine.connect() as con:
        con.execute(query)
        rs = con.execute(query2)

    for row in rs:
        print(row)


def _preprocessing():
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = "SELECT user_id, movie_id, rating FROM data"
    df = pd.read_sql(query, con=engine)

    train_df, val_df, test_df = split_data(df)

    # Save to SQL
    train_df.to_sql("train_data", engine, if_exists="replace", index=False)
    val_df.to_sql("val_data", engine, if_exists="replace", index=False)
    test_df.to_sql("test_data", engine, if_exists="replace", index=False)

    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")


def _train(**context):
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    train_query = "SELECT user_id, movie_id, rating FROM train_data"
    train_df = pd.read_sql(train_query, con=engine)

    val_query = "SELECT user_id, movie_id, rating FROM val_data"
    val_df = pd.read_sql(val_query, con=engine)

    model = train_model(train_df, val_df)

    run_id = context["run_id"]
    file_name = f"two_tower_model_{run_id}.pkl"

    model_path = f"/opt/models/{file_name}"
    model.save_model(model_path)


def _evaluate(**context):
    run_id = context["run_id"]
    model_path = f"/opt/models/two_tower_model_{run_id}.pkl"
    loaded_model = TwoTowerTripletNN.load_model(model_path)

    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = "SELECT user_id, movie_id, rating FROM test_data"
    test_df = pd.read_sql(query, con=engine)

    with open("/opt/models/user_encoder.pkl", "rb") as f:
        user_encoder = pickle.load(f)
    with open("/opt/models/movie_encoder.pkl", "rb") as f:
        movie_encoder = pickle.load(f)

    test_df["user_encoded"] = user_encoder.transform(test_df["user_id"])
    test_df["movie_encoded"] = movie_encoder.transform(test_df["movie_id"])

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    k = 10
    ndcg_scores = []

    all_movie_ids = torch.tensor(test_df["movie_encoded"].unique()).to(device)

    # Group by user
    for user_id, group in test_df.groupby("user_encoded"):
        if len(group) < k:
            continue  # skip users with fewer than k ratings

        # 1. Candidate movies (all unique test movies)
        candidate_movie_ids = all_movie_ids

        # 2. Repeat user ID to match candidate movie count
        user_tensor = torch.tensor([user_id] * len(candidate_movie_ids)).to(device)

        # 3. Predict scores
        scores = predict(loaded_model, user_tensor, candidate_movie_ids, device=device)

        # 4. Rank movie IDs by predicted score
        top_indices = torch.topk(scores, k).indices
        top_movie_ids = candidate_movie_ids[top_indices].cpu().numpy()

        # 5. Get actual relevance (binary: 1 if in top k for the user in test data, else 0)
        # Create a binary relevance list: 1 if movie was actually rated by this user
        # relevant_movies = set(group["movie_encoded"].values)
        # relevance_scores = [
        #     1 if movie_id in relevant_movies else 0 for movie_id in top_movie_ids
        # ]
        # 5. Create a dict of {movie_id: rating} for the user
        user_ratings_dict = dict(zip(group["movie_encoded"], group["rating"]))

        # 6. Relevance scores = actual ratings for the top-k predicted movies (0 if unrated)
        relevance_scores = [
            user_ratings_dict.get(int(movie_id), 0.0) for movie_id in top_movie_ids
        ]

        # 6. Compute NDCG for this user
        score = ndcg_score(relevance_scores, k)
        ndcg_scores.append(score)

    # Average NDCG across users
    mean_ndcg = np.mean(ndcg_scores)
    print(f"Mean NDCG@{k}: {mean_ndcg:.4f}")

    with engine.connect() as con:
        insert_query = """
            INSERT INTO model_metrics (timestamp, metric_name, metric_value, run_id)
            VALUES (%s, %s, %s, %s)
        """
        con.execute(
            insert_query,
            (datetime.now(datetime.timezone.utc), "nDCG", mean_ndcg, run_id),
        )
    ## TODO: change to if
    try:
        query_best_metric = "SELECT metric_value FROM best_score"
        best_score = pd.read_sql(query_best_metric, con=engine)
    except:
        with engine.connect() as con:
            insert_query = """
                INSERT INTO best_score (timestamp, metric_name, metric_value, run_id)
                VALUES (%s, %s, %s)
            """
            con.execute(
                insert_query,
                (datetime.now(datetime.timezone.utc), "nDCG", mean_ndcg, run_id),
            )


def _compare_metric_and_update_model(**context):
    run_id = context["run_id"]
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    # query = "SELECT user_id, movie_id FROM data"
    # X = pd.read_sql(query, con=engine)

    query_best_metric = "SELECT metric_value FROM best_score"
    best_score = pd.read_sql(query_best_metric, con=engine)

    query_current_metric = f""" SELECT * FROM model_metrics WHERE run_id='{run_id}' """
    current_score = pd.read_sql(query_current_metric, con=engine)

    if current_score > best_score:
        print(
            f"New best model found (run_id={run_id}) with metric {current_score:.4f} > {best_score:.4f}"
        )
        with engine.begin() as con:
            # Delete existing best score if it exists
            con.execute("DELETE FROM best_score")

            # Insert new best score
            insert_query = """
                INSERT INTO best_score (timestamp, metric_name, metric_value, run_id)
                VALUES (%s, %s, %s, %s)
            """
            con.execute(
                insert_query,
                (datetime.now(timezone.utc), "nDCG", current_score, run_id),
            )

        model_path = f"/opt/models/two_tower_model_{run_id}.pkl"
        model = TwoTowerTripletNN.load_model(model_path)
        model.eval()

        with open("/opt/models/user_encoder.pkl", "rb") as f:
            user_encoder = pickle.load(f)
        with open("/opt/models/movie_encoder.pkl", "rb") as f:
            movie_encoder = pickle.load(f)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        timestamp = datetime.datetime.now(datetime.timezone.utc)

        # Generate user embeddings
        user_ids = list(user_encoder.classes_)
        user_encoded = torch.tensor(user_encoder.transform(user_ids)).to(device)

        with torch.no_grad():
            user_emb = model.user_embedding(user_encoded)
            user_vec = model.user_fc(user_emb).cpu().numpy()

        user_rows = [
            (uid, emb.tolist(), run_id, timestamp)
            for uid, emb in zip(user_ids, user_vec)
        ]
        user_embeddings_df = pd.DataFrame(
            user_rows, columns=["user_id", "embedding", "run_id", "timestamp"]
        )

        # Generate movie embeddings
        movie_ids = list(movie_encoder.classes_)
        movie_encoded = torch.tensor(movie_encoder.transform(movie_ids)).to(device)

        with torch.no_grad():
            movie_emb = model.movie_embedding(movie_encoded)
            movie_vec = model.movie_fc(movie_emb).cpu().numpy()

        movie_rows = [
            (mid, emb.tolist(), run_id, timestamp)
            for mid, emb in zip(movie_ids, movie_vec)
        ]
        movie_embeddings_df = pd.DataFrame(
            movie_rows, columns=["movie_id", "embedding", "run_id", "timestamp"]
        )

        #  Save to DB

        movie_embeddings_df.to_sql(
            "movie_embeddings", engine, if_exists="replace", index=False
        )
        user_embeddings_df.to_sql(
            "user_embeddings", engine, if_exists="replace", index=False
        )

        # with engine.begin() as con:
        # TODO: Update best_score table
        # con.execute("UPDATE best_score SET metric_value = %s", (current_score,))

        # # Clear existing entries for this run_id to avoid duplicates
        # con.execute("DELETE FROM user_embeddings WHERE run_id = %s", (run_id,))
        # con.execute("DELETE FROM movie_embeddings WHERE run_id = %s", (run_id,))

        # # Insert embeddings
        # con.execute(
        #     """
        #     INSERT INTO user_embeddings (user_id, embedding, run_id, timestamp)
        #     VALUES %s
        # """,
        #     user_rows,
        # )

        # con.execute(
        #     """
        #     INSERT INTO movie_embeddings (movie_id, embedding, run_id, timestamp)
        #     VALUES %s
        # """,
        #     movie_rows,
        # )
    else:
        print(
            f"No improvement: current model (metric={current_score:.4f}) ≤ best (metric={best_score:.4f})"
        )


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "catchup": False,  # only the lateset non-triggered diagram will be automatically triggered
}

with DAG(
    "movie_recommendation_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 30),
) as dag:

    ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=_data_ingestion,
        dag=dag,
    )

    create_movie_ranking_table = PythonOperator(
        task_id="create_movie_ranking_table",
        python_callable=_create_movie_ranking_table,
        dag=dag,
    )

    create_watching_list_table = PythonOperator(
        task_id="create_watching_list_table",
        python_callable=_create_watching_list_table,
        dag=dag,
    )

    training_task = PythonOperator(
        task_id="train",
        python_callable=_train,
        dag=dag,
    )

    evaluation_task = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate,
        dag=dag,
    )

    compare_metric_and_update_model_task = PythonOperator(
        task_id="compare_and_update",
        python_callable=_compare_metric_and_update_model,
        dag=dag,
    )

    # Define the order of execution
    (
        ingestion_task
        >> [create_movie_ranking_table, create_watching_list_table]
        >> training_task
        >> evaluation_task
        >> compare_metric_and_update_model_task
    )
