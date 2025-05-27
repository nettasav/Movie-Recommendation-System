-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- CREATE DATABASE movies;

CREATE TABLE data (
	"user_id" BIGINT, 
	"movie_id" BIGINT, 
	"rating" INT, 
	"timestamp" BIGINT
    );

COPY data
FROM '/docker-entrypoint-initdb.d/u.data';
-- DELIMITER '\t'
-- CSV;

CREATE TABLE item (
	"movie_id" BIGINT, 
	"movie_title" TEXT, 
	"release_date" TEXT, 
	"video_release_date" TEXT, 
	"IMDb_URL" TEXT, 
	"unknown" BOOLEAN, 
	"Action" BOOLEAN, 
	"Adventure" BOOLEAN, 
	"Animation" BOOLEAN, 
	"Children's" BOOLEAN, 
	"Comedy" BOOLEAN, 
	"Crime" BOOLEAN, 
	"Documentary" BOOLEAN, 
	"Drama" BOOLEAN, 
	"Fantasy" BOOLEAN, 
	"Film-Noir" BOOLEAN, 
	"Horror" BOOLEAN, 
	"Musical" BOOLEAN, 
	"Mystery" BOOLEAN, 
	"Romance" BOOLEAN, 
	"Sci-Fi" BOOLEAN, 
	"Thriller" BOOLEAN, 
	"War" BOOLEAN, 
	"Western" BOOLEAN
);

COPY item
FROM '/docker-entrypoint-initdb.d/u.item'
DELIMITER '|'
ENCODING 'ISO-8859-1';
-- HEADER;

CREATE TABLE users (
	"user_id" BIGINT, 
	"age" BIGINT, 
	"gender" TEXT, 
	"occupation" TEXT, 
	"zip code" TEXT
);

COPY users
FROM '/docker-entrypoint-initdb.d/u.user'
DELIMITER '|';


CREATE TABLE info (
	"0" TEXT
);

COPY info
FROM '/docker-entrypoint-initdb.d/u.info';

CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name TEXT,
    metric_value FLOAT,
	run_id TEXT
);

CREATE TABLE IF NOT EXISTS best_score (
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metric_name TEXT,
    metric_value FLOAT,
	run_id TEXT PRIMARY KEY
);
-- CREATE TABLE genre (
-- 	"0" TEXT, 
-- 	"1" BIGINT
-- );

-- COPY genre
-- FROM '/docker-entrypoint-initdb.d/u.genre'
-- DELIMITER '|';

CREATE TABLE IF NOT EXISTS train_data (
    userId BIGINT,
    movieId BIGINT,
    rating INT
);

CREATE TABLE IF NOT EXISTS val_data (
    userId BIGINT,
    movieId BIGINT,
    rating INT
);

CREATE TABLE IF NOT EXISTS test_data (
    userId BIGINT,
    movieId BIGINT,
    rating INT
);

CREATE TABLE user_embeddings (
    user_id TEXT PRIMARY KEY,
    embedding vector(64),
    run_id TEXT,
    timestamp TIMESTAMPTZ
);

CREATE TABLE movie_embeddings (
    movie_id TEXT PRIMARY KEY,
    embedding vector(64),
    run_id TEXT,
    timestamp TIMESTAMPTZ
);