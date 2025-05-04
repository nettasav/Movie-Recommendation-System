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


@bp.route("/predict/popular")
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
