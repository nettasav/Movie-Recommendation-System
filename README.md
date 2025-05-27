# Movie-Recommendation-System

# WIP

# 🎬 Movie Recommendation System – Phase 3: Two-Tower Neural Network

## 📌 Overview

This branch implements a movie recommendation system utilizing a Two-Tower Neural Network architecture, integrated with PostgreSQL enhanced by the pgvector extension. The system efficiently handles large-scale datasets and provides personalized movie recommendations by learning user and item embeddings separately, leveraging vector similarity search directly within the database.



## 🧠 Phase 3: Two-Tower Neural Network Based Recommendation

The Two-Tower Neural Network (TTNN) is a deep learning architecture designed to generate personalized recommendations by independently learning dense vector representations (embeddings) for users and items. Unlike the Factorization Machines model in Phase 2, which relied on hand-crafted features and linear interactions, this phase leverages neural networks and vector similarity to capture more complex patterns in user behavior. Both user and item embeddings are stored in a PostgreSQL database enhanced with the pgvector extension to enable efficient similarity search. A Flask API serves recommendations to end users.

The two-tower model consists of:

- **User Tower:** Processes user-related features to generate user embeddings.

- **Item Tower:** Processes item-related features to generate item embeddings.

The similarity between user and item embeddings is computed to predict user preferences.

### 🔑 Key Characteristics:

- **Personalization:**  Learns separate embedding spaces for users and items, enabling highly tailored recommendations.

- **Scalability:**  Precomputing and storing embeddings in the database allows fast and scalable inference via vector similarity.

- **Ranking-Aware Evaluation: Performance is evaluated using the metrics Normalized Discounted Cumulative Gain (NDCG) metric to ensure quality of ranked lists.

- **Cold Start:**  If the user is not found in the embedding table, the system falls back to the popularity-based recommendations from Phase 1.

- **End-to-End Workflow:**  Integrates data preprocessing, model training, evaluation, embedding storage, and deployment through Docker and Flask.

## 🔄 Data Pipeline Workflow
The data pipeline is orchestrated using Apache Airflow and automates the following steps:

- **Data Ingestion:** Airflow DAGs are scheduled to ingest new user-movie rating data into the PostgreSQL database.

- **Model Training:** The two_tower_train.py script is triggered to train the Two-Tower Neural Network. It learns separate embeddings for users and movies.

- **Embedding Storage:** After training, user and movie embeddings are stored in the PostgreSQL database using the pgvector extension to support fast similarity search.

- **Model Evaluation:** The model is evaluated using the NDCG metric to assess recommendation ranking quality. Metrics are saved to the model_metrics table.

- **Model Comparison and Update:**

 - The best score is stored in the best_score table.

 - After each run, the latest score is compared against the best score to determine whether to update the stored embeddings.

 - If the current model outperforms the previous one, the embeddings in the database are updated for serving live recommendations.

## 🚀 Getting Started
### Prerequisites
- Docker

- Docker Compose

### Installation
1. Clone the repository:

bash 
```
git clone https://github.com/nettasav/Movie-Recommendation-System.git
cd Movie-Recommendation-System
git checkout phase3-two-tower
```
2. Build and start the containers:

bash
```
docker-compose up --build
```
3. Access the services:

- Airflow UI: http://localhost:8000

- API Endpoint: http://localhost:5000/recommend/two_tower_NN?user_id=123

## 📈 API Usage
Once the API is running, you can access the Two-Tower Neural Network-based recommendations:

- **Endpoint:**

```
http://localhost:5000/recommend/two_tower_NN?user_id=<USER_ID>

```
- **Description:**
Returns the top 10 recommended movie titles for a given user using vector similarity search between user and movie embeddings stored in PostgreSQL with pgvector.

- **Query Parameter:** user_id (integer): The ID of the user to generate personalized recommendations for

- **Method:** GET

- **Example Request:**

```
curl "http://localhost:5000/recommend/two_tower_NN?user_id=123"
```

- **Example Response:**

```
[
  "The Matrix",
  "Inception",
  "The Lord of the Rings: The Fellowship of the Ring",
  "Interstellar",
  "Fight Club",
  "The Dark Knight",
  "Pulp Fiction",
  "Forrest Gump",
  "Gladiator",
  "The Shawshank Redemption"
]
```
## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.








