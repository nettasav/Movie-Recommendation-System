from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from random import randint
import pandas as pd
from datetime import datetime
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
        rs = con.execute(query2) # Create logs

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



def _train(**context):
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = "SELECT user_id, movie_id, rating FROM data"
    df = pd.read_sql(query, con=engine)

    # Encode userId and movieId
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df['userId'] = user_encoder.fit_transform(df['userId'])
    df['movieId'] = movie_encoder.fit_transform(df['movieId'])

    num_users = df['userId'].nunique()
    num_movies = df['movieId'].nunique()
    
    unique_users = df['userId'].unique()
    train_users, test_users = train_test_split(unique_users, test_size=0.2, random_state=42)
    train_users, val_users = train_test_split(train_users, test_size=0.1, random_state=42)

    train_df = df[df['userId'].isin(train_users)]
    val_df = df[df['userId'].isin(val_users)]
    test_df = df[df['userId'].isin(test_users)]
    
    # train_triplets = generate_triplets(train_df)
    # val_triplets = generate_triplets(val_df)
    # test_triplets = generate_triplets(test_df)

    train_dataset = MovieLensTripletDataset(train_df)
    val_dataset = MovieLensTripletDataset(val_df)
    test_dataset = MovieLensTripletDataset(test_df)

    BATCH_SIZE = 256

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


    model = TwoTowerTripletNN(num_users, num_movies, embedding_dim=64)
    criterion = TripletLoss(margin=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

    train_losses, val_losses = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=10,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

    # Save to database
    # train_df.to_sql("train_data", engine, if_exists="replace", index=False)
    test_df.to_sql("test_data_two_tower", engine, if_exists="replace", index=False)

    # print("Leave-5-Out split completed and saved.")

    
    
    # engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    # query = """
    #         SELECT user_id, movie_id, rating
    #         FROM data
    #         """
    # with engine.connect() as con:
    #     con.execute(query)
    #     rs = con.execute(query)

    # df = pd.read_sql(query, engine)

    X = train_df[["user_id", "movie_id"]].values
    y = train_df["rating"].values

    fm_model = FactorizationMachine(X, y, k=20, lambda_L2=0.001, learning_rate=0.001, batch_size=128
        )
    fm_model.fit(num_epochs=30, patience=2, tol=1e-6, print_cost=True)
    
    # current_datetime = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_id = context["run_id"]
    # file_name = f"fm_model_{current_datetime}.pkl"
    file_name = f"fm_model_{run_id}.pkl"

    model_path = f"/opt/models/{file_name}"
    fm_model.save_model(model_path)

    # y_pred = fm_model.predict(X)

    # df_predictions = df[["user_id", "movie_id"]].copy()
    # df_predictions["predicted_rating"] = y_pred

    # df_predictions.to_sql("predicted_ratings", engine, if_exists="append", index=False)

def _evaluate(**context):
    run_id = context["run_id"]
    model_path = f"/opt/models/fm_model_{run_id}.pkl"
    loaded_model = FactorizationMachine.load_model(model_path)

    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = "SELECT user_id, movie_id, rating FROM test_data"
    test_df  = pd.read_sql(query, con=engine)

    y_true = test_df ['rating'].values
    X_test = test_df [["user_id", "movie_id"]].values

    y_pred = loaded_model.predict(X_test)
    
    test_df["predicted"] = y_pred
    test_df["true"] = y_true

    ndcg_scores = []
    for user_id, group in test_df.groupby("user_id"):
        if len(group) < 2:
            continue  # skip if there's nothing to rank

        # Sort by predicted ratings
        y_true_group = group.sort_values("predicted", ascending=False)["true"].values.reshape(1, -1)
        ideal_group = group.sort_values("true", ascending=False)["true"].values.reshape(1, -1)

        ndcg = ndcg_score(ideal_group, y_true_group)
        ndcg_scores.append(ndcg)

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    print(f"Mean nDCG@{len(group)}: {mean_ndcg:.4f}")

    with engine.connect() as con:
        insert_query = """
            INSERT INTO model_metrics (timestamp, metric_name, metric_value, run_id)
            VALUES (%s, %s, %s)
        """
        con.execute(insert_query, (datetime.now(datetime.timezone.utc), "nDCG", mean_ndcg, run_id))
    try:
        query_best_metric = "SELECT metric_value FROM best_score"
        best_score = pd.read_sql(query_best_metric, con=engine)
    except:
        with engine.connect() as con:
            insert_query = """
                INSERT INTO best_score (timestamp, metric_name, metric_value, run_id)
                VALUES (%s, %s, %s)
            """
            con.execute(insert_query, (datetime.now(datetime.timezone.utc), "nDCG", mean_ndcg, run_id))
       
def _compare_metric_and_update_model(**context):
    run_id = context["run_id"]
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = "SELECT user_id, movie_id FROM data"
    X = pd.read_sql(query, con=engine)
   
    query_best_metric = "SELECT metric_value FROM best_score"
    best_score = pd.read_sql(query_best_metric, con=engine)

    query_current_metric = f""" SELECT * FROM model_metrics WHERE run_id='{run_id}' """
    current_score = pd.read_sql(query_current_metric, con=engine)

    if current_score > best_score:
        model_path = f"/opt/models/fm_model_{run_id}.pkl"
        loaded_model = FactorizationMachine.load_model(model_path)

        y_pred = loaded_model.predict(X)

        df_predictions = X[["user_id", "movie_id"]].copy()
        df_predictions["predicted_rating"] = y_pred

        df_predictions.to_sql("predicted_ratings", engine, if_exists="replace", index=False)
    
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
        dag=dag
    )
   
    # Define the order of execution
    (
        ingestion_task
        >> [create_movie_ranking_table, create_watching_list_table]
        >> training_task
        >> evaluation_task
        >> compare_metric_and_update_model_task
    )
