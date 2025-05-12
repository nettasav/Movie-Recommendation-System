# 🎬 Movie Recommendation System – Phase 2: Factorization Machines
Welcome to the Movie Recommendation System project! This repository showcases a modular and extensible framework for building movie recommendation engines. Each branch represents a distinct recommendation strategy. This Phase 2: Factorization Machines branch focuses on a Factorization Machines (FMs)-based recommendation approach, designing to provide personalized movie recommendations based on user preferences and historical data.

## 🧠 Phase 2: Factorization Machines Based Recommendation
This project implements a movie recommendation system using Factorization Machines (FMs), a powerful model for capturing interactions between variables in sparse datasets. Unlike the baseline model in Phase 1, which provided the same recommendations to all users, this phase captures user-item interactions to tailor suggestions to individual preferences. The entire workflow is orchestrated using Apache Airflow, with data stored in a PostgreSQL database. A **Flask API** serves the recommendations to end-users.

**Key Characteristics:**
- **Personalization:** Learns user-specific preferences for improved recommendations
- **Scalability:** Efficient training via Mini-Batch Gradient Descent enables handling large datasets
- **Ranking-Aware Evaluation:** Uses **Normalized Discounted Cumulative Gain (NDCG)** to measure the quality of ranked recommendations
- **End-to-End Workflow:** Incorporates automated data ingestion, model training, evaluation, and deployment via **Airflow** and **Docker**


## 🔄 Data Pipeline Workflow
The data pipeline is orchestrated using **Apache Airflow** and follows these steps:
1. **Data Ingestion:** Airflow DAGs are scheduled to ingest new user-movie rating data into the PostgreSQL database
2. **Model Training:** The fm_model_MBGD.py script is triggered to train the FM model using MBGD
3. **Model Evaluation** Post-training, the model's performance is evaluated using the NDCG metric to assess the quality of recommendations
4. **Model Deployment:** The trained model is saved and loaded by the Flask API for serving predictions

## 🚀 Getting Started
### Prerequisites
- Docker
- Docker Compose

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/nettasav/Movie-Recommendation-System.git
cd Movie-Recommendation-System
git checkout phase2-factorization-machines
```

2. **Build and start the containers:**
```bash

docker-compose up --build
```

3. **Access the services:**
- **Airflow UI:** [http://localhost:8000](http://localhost:8000)
- **API Endpoint:** [http://localhost:5000](http://localhost:5000)


## 📈 API Usage
Once the API is running, you can access the FM-based recommendations:

- **Endpoint:** 
```bash
http://localhost:5000/recommend/fm?user_id=<USER_ID>
```

- **Description:** Returns top 5 recommended movie titles for a given user
- **Query Parameter:** `user_id` (integer): The ID of the user to generate recommendations for
- **Method:** `GET`
- **Example Request:**
```bash
curl "http://localhost:5000/recommend/fm?user_id=123"
```

- **Example Response:**
```json
{
  "recommendations": [
  "The Shawshank Redemption",
  "Forrest Gump",
  "Inception",
  "The Dark Knight",
  "Pulp Fiction"
  ]
}
```

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

