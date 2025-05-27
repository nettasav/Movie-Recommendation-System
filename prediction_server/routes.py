from flask import Blueprint, request, render_template
import pandas as pd
import sqlalchemy
from sqlalchemy import text

import psycopg2
from __init__ import engine

bp = Blueprint("routes", __name__)


@bp.route("/")
def main_page():
    return render_template("base.html")


@bp.route("/watched_movies")
def watched_movies():
    user_id = request.args.get("user_id", "")
    try:
        query = f"SELECT * FROM watching_list WHERE user_id={user_id};"
        with engine.connect() as con:
            res = con.execute(text(query)).mappings().all()
            print(f"result: {res}")
        return res
    except Exception as e:
        print("I'm in exception", e)
        return "user not found\n"


@bp.route("/recommend/popular")
def predict_movies():
    user_id = request.args.get("user_id", "")
    try:
        already_seen_movies_by_user_id = (
            f"select * from watching_list where user_id={user_id}"
        )
        watching_list = pd.read_sql(already_seen_movies_by_user_id, con=engine)
        num_of_movies = watching_list["num_watched"].iloc[0]
        limit_movies = num_of_movies + 5

        ratings_query = f"select movie_id,avg_rating, movie_title from ratings_view LIMIT {limit_movies}"

        ratings = pd.read_sql(ratings_query, con=engine)
        watched_movies_list = list(watching_list.watched_movies[0])

        top5 = ratings.loc[~ratings["movie_id"].isin(watched_movies_list)][:5]
        print(top5.movie_title)
        return top5.movie_title.tolist()
    except ValueError:
        return render_template("error.html", user_id=user_id), 400
    except Exception as e:
        print("I'm in exception", e)
        return render_template("error.html", user_id=user_id), 500


@bp.route("/recommend/two_tower_NN")
def predict_movies_two_tower():
    user_id = request.args.get("user_id", "")

    if not user_id.isdigit():
        return render_template("error.html", user_id=user_id), 400

    try:
        # Check if the user exists in user_embeddings
        is_exist_query = f"""
            SELECT 1
            FROM user_embeddings
            WHERE user_id = {user_id}
            LIMIT 1;
        """
        with engine.connect() as con:
            result = con.execute(text(is_exist_query)).fetchone()

        if not result:
            print(
                f"Cold start: user_id {user_id} not found in predicted_ratings. Using popular fallback."
            )
            ratings_query = (
                f"select movie_id,avg_rating, movie_title from ratings_view LIMIT 10"
            )
            top10_ratings = pd.read_sql(ratings_query, con=engine)
            return top10_ratings.movie_title.tolist()

        query = f"""
            SELECT
            m.movie_id,
            m.embedding as movie_emb,
            i.title AS movie_title,
            u.embedding >> m.embedding as score

            -- (SELECT u.user_id, u.user_embedding FROM user_embeddings u WHERE user_id = {user_id}) AS user_id, 
            -- (SELECT u.user_embedding FROM user_embeddings u WHERE user_id = {user_id}) AS user_embedding, 

            FROM movie_embeddings m
            JOIN item i ON i.movie_id = m.movie_id
            JOIN user_embeddings u ON u.user_id = {user_id}
            JOIN watching_list wl ON wl.user_id = u.user_id
            WHERE m.movie_id != ALL(wl.watched_movies)
            ORDER BY score DESC
            LIMIT 10; 
        """

        # query = f"""
        #     SELECT pr.movie_id, pr.predicted_rating, m.title AS movie_title
        #     FROM predicted_ratings pr
        #     JOIN item m ON pr.movie_id = m.movie_id
        #     JOIN watching_list wl ON pr.user_id = wl.user_id
        #     WHERE pr.movie_id != ALL(wl.watched_movies)
        #         AND pr.user_id = {user_id}
        #         -- AND pr.user_id = :user_id  ## if use the other read_sql form with params
        #     ORDER BY pr.predicted_rating DESC
        #     LIMIT 5;
        # """

        top_predictions = pd.read_sql(query, con)
        #  df = pd.read_sql(text(query), con, params={"user_id": user_id}) # also change line 69 if using this

        print(top_predictions.movie_title)

        return top_predictions["movie_title"].tolist()
    except Exception as e:
        print("Two-Tower NN recommendation error:", e)
        return render_template("error.html", user_id=user_id), 500
