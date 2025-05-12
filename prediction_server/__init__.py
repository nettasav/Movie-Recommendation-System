from flask import Flask, render_template
from sqlalchemy import create_engine
from prediction_server import routes


app = Flask(__name__)
engine = create_engine("postgresql://airflow:airflow@postgres_movies:5432/movies")


def page_not_found(e):
    print(e)
    return render_template("error.html"), 404


def create_app():
    app.register_blueprint(routes.bp)
    app.register_error_handler(404, page_not_found)
    return app
