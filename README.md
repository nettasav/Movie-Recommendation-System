# 🎬 Movie Recommendation System 
Welcome to the **Movie Recommendation System project!** This repository showcases a modular, scalable and extensible framework for building movie recommendation engines. Each branch represents a distinct recommendation strategy.


## 📁 Project Overview
This repository is structured into three distinct branches, each representing a different recommendation strategy:

1. Phase 1 – Popularity-Based Recommender
A straightforward approach that recommends movies based on the overall popularity. This phase serves as a fall back for the next phases tp deal with the *cold start* problem. 

2. Phase 2 – Factorization Machines
This phase employs Factorization Machines to capture complex interactions between users and items, enhancing recommendation accuracy.

3. Phase 3 – Two-Tower Neural Network
An advanced deep learning model that learns separate embeddings for users and items, enabling personalized recommendations through similarity in the embedding space.


## 🧰 Technologies & Tools
Across all phases, the project leverages a robust tech stack:

- **Data Orchestration:** Apache Airflow for scheduling and managing data pipelines.
- **Containerization:** Docker and Docker Compose for consistent development and deployment environments.
- **Database:** PostgreSQL for structured data storage and retrieval.
- **Model Serving:** Flask to expose machine learning models as RESTful APIs.
- **Machine Learning:** Python-based implementations of recommendation algorithms, including Factorization Machines and Two-Tower Neural Networks.


## 🚀 Getting Started
To explore a specific recommendation strategy:

1. Clone the Repository:
```bash
git clone https://github.com/nettasav/Movie-Recommendation-System.git
```
2. Checkout the Desired Branch:

``` bash 
git checkout phase1-popularity  # or phase2-factorization-machines, phase3-two-tower
```
3. Launch the Application:

``` bash
docker-compose up --build
```

This will set up all necessary services, including the database, Airflow scheduler, and the FastAPI prediction server.

## 📬 Contact
For questions, feedback, or collaboration opportunities, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/netta-savin/).

## 📄 License
This project is licensed under the MIT License. See the LICENSE file for details.

