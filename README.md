# 🎬 Movie Recommendation System – Phase 1: Popularity-Based Model
Welcome to the Movie Recommendation System project! This repository showcases a modular and extensible framework for building movie recommendation engines. Each branch represents a distinct recommendation strategy. This phase1-popularity branch focuses on a popularity-based recommendation approach, serving as a foundational baseline for more advanced models.

## 🧠 Phase 1: Popularity-Based Recommendation
In this initial phase, the system recommends movies that are popular among the general audience. This approach does not require user-specific data, making it ideal for new users or as a baseline for comparison with more personalized methods.

**Key Characteristics:**

- **Simplicity:** Easy to implement and interpret
- **No Personalization:** Recommendations are the same for all users
- **Baseline Benchmark:** Serves as a reference point to evaluate more complex models

## 🔄 Data Pipeline Workflow
The data pipeline is orchestrated using Apache Airflow and follows these steps:
1. **Data Ingestion:** Fetches movie data from external sources or local files
2. **Feature Engineering:** Calculates popularity metrics (e.g., average rating, rating count)
3. **Data Storage: Stores** the processed data in a structured format for quick retrieval


## 🚀 Getting Started
### Prerequisites
- Docker
- Docker Compose

### Installation
1. **Clone the repository:**
```bash
git clone https://github.com/nettasav/Movie-Recommendation-System.git
cd Movie-Recommendation-System
git checkout phase1-popularity
```

2. **Build and start the containers:**
```bash

docker-compose up --build
```

3. **Access the services:**
- **Airflow UI:** [http://localhost:8000](http://localhost:8000)
- **API Endpoint:** [http://localhost:5000](http://localhost:5000)


## 📈 API Usage
Once the API is running, you can access the popularity-based recommendations:

- **Endpoint:** 
``` bash
http://localhost:5000/recommend/popular?user_id=<USER_ID>
```
- **Query Parameter:** user_id (integer): The ID of the user to generate recommendations for
- **Method:** `GET`
- **Example Request:**
``` bash
curl http://localhost:5000/recommend/popular?user_id=<USER_ID>
```
- **Example Response:**
```json
[
  "Inception",
  "Pulp Fiction",
  "The Dark Knight",
  "The Matrix",
  "Fight Club"
]
```
- **Output Format:** A JSON array of top 5 movie titles (list[str]).
- **Error Handling:**
- Returns an HTML error page if user_id is invalid or something fails internally
- Status code: 400 or 500

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

