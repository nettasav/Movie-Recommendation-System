from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta
from random import randint
import time
import pandas as pd

from sqlalchemy import create_engine
import psycopg2
import logging


from fm_model_MBGD import FactorizationMachine

def _data_ingestion():
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

def _train():
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = """
            SELECT user_id, movie_id, rating
            FROM data
            """
    with engine.connect() as con:
        con.execute(query)
        rs = con.execute(query)

    df = pd.read_sql(query, engine)

    X = df[["user_id", "movie_id"]].values
    y = df["rating"].values

    fm_model = FactorizationMachine(X, y, k=20, lambda_L2=0.001, learning_rate=0.001, batch_size=128
        )
    fm_model.fit(num_epochs=30, patience=2, tol=1e-6, print_cost=True)
    y_pred = fm_model.predict(X)

    df_predictions = df[["user_id", "movie_id"]].copy()
    df_predictions["predicted_rating"] = y_pred

    df_predictions.to_sql("predicted_ratings", engine, if_exists="append", index=False)


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

    # Define the order of execution
    (
        ingestion_task
        >> [create_movie_ranking_table, create_watching_list_table]
        >> training_task
    )
