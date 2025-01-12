# Recommendation Systems: From Basics to Deep Neural Networks

## Overview
This project implements various recommendation system techniques ranging from basic similarity-based methods to advanced deep neural network models. We explore how each method performs on the **2023 Amazon Reviews Dataset**, and evaluate them using standard metrics like **Hit Rate (HR)**, **Normalized Discounted Cumulative Gain (NDCG)**, **Precision**, and **Recall**.

## Authors
- 🔹 **Elmimouni Zakarya** – ENSAE Paris (zakarya.elmimouni@ensae.fr)
- 🔹 **Khairaldin Ahmed** – ENSAE Paris (ahmed.khairaldin@ensae.fr)
- 🔹 **Razig Amine** – ENSAE Paris (amine.razig@ensae.fr)

## Table of Contents
1. [Objective](#objective)
2. [Dataset](#dataset)
3. [Techniques Explored](#techniques-explored)
4. [Evaluation Procedure](#evaluation-procedure)
5. [How to Run the Project](#how-to-run-the-project)
6. [Requirements](#requirements)
7. [License](#license)

## Objective
The primary objective of this project is to explore and implement different recommendation system techniques, analyze their performance, and identify the most effective ones for generating high-quality recommendations.

### Evaluation
We use the **Leave-2-Out procedure** to evaluate models, where two positive interactions are reserved for testing per user. We add 99 random negative samples to simulate realistic recommendation scenarios.

Performance is assessed through the following metrics:
- **Hit Rate (HR)**
- **Normalized Discounted Cumulative Gain (NDCG)**
- **Precision**
- **Recall**

## 📊 Dataset
The **2023 Amazon Reviews Dataset** is used, which contains user-item interaction data, including ratings and timestamps. It provides a rich foundation for testing collaborative filtering models and comparing various recommendation techniques.

- **Source**: [Amazon Reviews Dataset](https://www.amazon.com/)
- **Details**: User-item interaction data including ratings and timestamps.

## Techniques Explored

1. **Cosine Similarity-Based Recommendation**  
   A basic approach that calculates the cosine similarity between user or item vectors to make recommendations based on previously interacted items.

2. **Singular Value Decomposition (SVD)**  
   Decomposes the user-item interaction matrix to extract latent factors, capturing the underlying preferences and item features.

3. **Matrix Factorization via SGD**  
   An extension of SVD where latent user-item factors are learned using **Stochastic Gradient Descent (SGD)**.

4. **Binary Conversion of Ratings**  
   Transforms the problem into binary classification, marking items as relevant or irrelevant for easier prediction.

5. **Neural Matrix Factorization**  
   A neural network-based model that models complex user-item interactions using deep learning techniques.

6. **Multilayer Perceptron (MLP)**  
   Deep learning model using multiple layers of neurons to capture complex patterns in user-item relationships.

7. **Neural Collaborative Filtering (NCF)**  
   A hybrid approach combining **MLP** and **latent factor models** to learn complex, high-order interactions between users and items.

## Evaluation Procedure

### Leave-2-Out Procedure
For each user, two interactions are kept for testing, and 99 random negative samples are added to simulate real-world recommendation conditions.

### 📏 Metrics Used
- **Hit Rate (HR)**: Whether relevant items are present in the top \( K \) recommendations.
- **Normalized Discounted Cumulative Gain (NDCG)**: Measures ranking quality by considering the positions of relevant items in the recommended list.
- **Precision**: The fraction of recommended items that are relevant.
- **Recall**: The fraction of relevant items that are successfully recommended.

## How to Run the Project

### Step 1: Clone the Repository
```bash
git clone https://github.com/arazig/Advanced-ML-project.git
cd Advanced-ML-project
```

### 💻 Step 2: Install Required Libraries
Make sure you have **Python 3.8+** installed.

Install the required dependencies by running:
```bash
pip install -r requirements.txt
```

### Step 3: Run the Code
You can run the main scripts for each recommendation method (e.g., `cosine_similarity.py`, `svd.py`, etc.) to test the models individually or run the entire pipeline. For example:
```bash
TODO
```

### Step 4: Evaluate Models
After running the models, use the evaluation script to calculate **HR**, **NDCG**, **Precision**, and **Recall** on the dataset.

### Step 5: Results
The results of the different models' performance can be found in the `results/` directory.

## Requirements
- **Python 3.8+**
- Libraries:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scikit-learn`
  - `pytorch`
  - `torch`
- CPU or **GPU** for faster training on neural models.
