from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from datetime import datetime, timedelta, timezone
from random import randint
import time
import pandas as pd
from datetime import datetime
from sklearn.metrics import ndcg_score

import sqlalchemy
from sqlalchemy import create_engine

import psycopg2
import logging

import sys
import os

# Get the absolute path of the directory containing the current DAG file
dag_folder = os.path.dirname(os.path.abspath(__file__))

# Get the parent directory (which is the root of your project)
src_folder = os.path.dirname(dag_folder)

# Add the project root to the Python path
if src_folder not in sys.path:
    sys.path.append(src_folder)


from src.fm_model_MBGD import FactorizationMachine


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


def _train(**context):
    engine = sqlalchemy.create_engine(
        "postgresql://airflow:airflow@postgres_movies:5432/movies"
    )
    query = "SELECT user_id, movie_id, rating FROM data"

    # df = pd.read_sql(query, con=engine)
    # # with engine.connect() as con:
    # #     # Pass the active connection object 'con' to pd.read_sql
    # #     df = pd.read_sql(query, con=con)

    # N = 5  # Leave-N-Out

    # train_rows = []
    # test_rows = []

    # for user_id, user_df in df.groupby("user_id"):
    #     if len(user_df) <= N:
    #         train_rows.append(user_df)
    #         continue
    #     user_df = user_df.sort_values(
    #         by="movie_id", ascending=True
    #     )  # stable & deterministic
    #     test_rows.append(user_df.iloc[-N:])  # last N for test
    #     train_rows.append(user_df.iloc[:-N])  # rest for training

    # train_df = pd.concat(train_rows)
    # test_df = pd.concat(test_rows)

    # # Save to database
    # # train_df.to_sql("train_data", engine, if_exists="replace", index=False)
    # test_df.to_sql("test_data", engine, if_exists="replace", index=False)

    with engine.connect() as con:  # conn_obj is a SQLAlchemy Connection object
        # Access the raw, underlying DBAPI connection
        # This is typically a psycopg2 connection for PostgreSQL
        raw_dbapi_connection = con.connection

        # Pass the raw DBAPI connection to pandas
        # This object *will* have the .cursor() method that pandas is looking for
        df = pd.read_sql(query, con=raw_dbapi_connection)

        N = 5  # Leave-N-Out

        train_rows = []
        test_rows = []

        for user_id, user_df in df.groupby("user_id"):
            if len(user_df) <= N:
                train_rows.append(user_df)
                continue
            user_df = user_df.sort_values(
                by="movie_id", ascending=True
            )  # stable & deterministic
            test_rows.append(user_df.iloc[-N:])  # last N for test
            train_rows.append(user_df.iloc[:-N])  # rest for training

        train_df = pd.concat(train_rows)
        test_df = pd.concat(test_rows)

        # Save to database using the raw DBAPI connection as well
        # train_df.to_sql("train_data", con=raw_dbapi_connection, if_exists="replace", index=False)
        test_df.to_sql("test_data", con=con, if_exists="replace", index=False)
    # The SQLAlchemy connection (conn_obj) is closed here,
    # which also handles closing the raw_dbapi_connection.
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

    fm_model = FactorizationMachine(
        X, y, k=20, lambda_L2=0.001, learning_rate=0.001, batch_size=128
    )
    fm_model.fit(X, y, num_epochs=1, patience=2, tol=1e-6, print_cost=True)
    # current_datetime = str(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    run_id = context["run_id"]
    # file_name = f"fm_model_{current_datetime}.pkl"
    file_name = f"fm_model_{run_id}.pkl"

    model_path = f"/opt/airflow/models/{file_name}"
    fm_model.save_model(model_path)

    # y_pred = fm_model.predict(X)

    # df_predictions = df[["user_id", "movie_id"]].copy()
    # df_predictions["predicted_rating"] = y_pred

    # df_predictions.to_sql("predicted_ratings", engine, if_exists="append", index=False)


def _evaluate(**context):
    run_id = context["run_id"]
    model_path = f"/opt/airflow/models/fm_model_{run_id}.pkl"
    loaded_model = FactorizationMachine.load_model(model_path)

    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")
    query = "SELECT user_id, movie_id, rating FROM test_data"
    test_df = pd.read_sql(query, con=engine)

    y_true = test_df["rating"].values
    X_test = test_df[["user_id", "movie_id"]].values

    y_pred = loaded_model.predict(X_test)

    test_df["predicted"] = y_pred
    test_df["true"] = y_true

    ndcg_scores = []
    for user_id, group in test_df.groupby("user_id"):
        if len(group) < 2:
            continue  # skip if there's nothing to rank

        # Sort by predicted ratings
        y_true_group = group.sort_values("predicted", ascending=False)[
            "true"
        ].values.reshape(1, -1)
        ideal_group = group.sort_values("true", ascending=False)["true"].values.reshape(
            1, -1
        )

        ndcg = ndcg_score(ideal_group, y_true_group)
        ndcg_scores.append(ndcg)

    mean_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    print(f"Mean nDCG@{len(group)}: {mean_ndcg:.4f}")

    with engine.begin() as con:
        insert_query = """
            INSERT INTO model_metrics (timestamp, metric_name, metric_value, run_id)
            VALUES (%s, %s, %s)
        """
        con.execute(
            insert_query, (datetime.now(timezone.utc), "nDCG", mean_ndcg, run_id)
        )
    try:
        query_best_metric = "SELECT metric_value FROM best_score"
        best_score = pd.read_sql(query_best_metric, con=engine)
    except:
        with engine.begin() as con:
            insert_query = """
                INSERT INTO best_score (timestamp, metric_name, metric_value, run_id)
                VALUES (%s, %s, %s)
            """
            con.execute(
                insert_query, (datetime.now(timezone.utc), "nDCG", mean_ndcg, run_id)
            )


def _compare_metric_and_update_model(**context):
    run_id = context["run_id"]
    engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")

    # Get full data for predicting if needed
    query = "SELECT user_id, movie_id FROM data"
    X = pd.read_sql(query, con=engine)

    # Get current score
    query_current_metric = f""" SELECT * FROM model_metrics WHERE run_id='{run_id}' """
    current_score_df = pd.read_sql(query_current_metric, con=engine)

    # Get best score
    query_best_metric = "SELECT * FROM best_score"
    best_score_df = pd.read_sql(query_best_metric, con=engine)

    current_score = current_score_df["metric_value"].iloc[0]
    best_score = best_score_df["metric_value"].iloc[0]

    if current_score > best_score:
        print(f"Updating best_score. Old: {best_score}, New: {current_score}")

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

        model_path = f"/opt/airflow/models/fm_model_{run_id}.pkl"
        loaded_model = FactorizationMachine.load_model(model_path)

        y_pred = loaded_model.predict(X.values)

        df_predictions = X.copy()
        df_predictions["predicted_rating"] = y_pred

        df_predictions.to_sql(
            "predicted_ratings", engine, if_exists="replace", index=False
        )
    else:
        print(
            f"Current score ({current_score:.4f}) is not better than best score ({best_score:.4f})"
        )


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    # "catchup": False,  # only the lateset non-triggered diagram will be automatically triggered
}

with DAG(
    "movie_recommendation_dag",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=datetime(2025, 1, 30),
    catchup=False,  # only the lateset non-triggered diagram will be automatically triggered
) as dag:

    ingestion_task = PythonOperator(
        task_id="data_ingestion",
        python_callable=_data_ingestion,
    )

    create_movie_ranking_table = PythonOperator(
        task_id="create_movie_ranking_table",
        python_callable=_create_movie_ranking_table,
    )

    create_watching_list_table = PythonOperator(
        task_id="create_watching_list_table",
        python_callable=_create_watching_list_table,
    )

    training_task = PythonOperator(
        task_id="train",
        python_callable=_train,
        provide_context=True,
    )

    evaluation_task = PythonOperator(
        task_id="evaluate",
        python_callable=_evaluate,
        provide_context=True,
    )

    compare_metric_and_update_model_task = PythonOperator(
        task_id="compare_and_update",
        python_callable=_compare_metric_and_update_model,
        provide_context=True,
    )

    # Define the order of execution
    (
        ingestion_task
        >> [create_movie_ranking_table, create_watching_list_table]
        >> training_task
        >> evaluation_task
        >> compare_metric_and_update_model_task
    )
