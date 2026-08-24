# 🚀 Customer Churn Prediction

### End-to-End Machine Learning Application using XGBoost, FastAPI & Interactive Web Interface

> A complete Machine Learning project that analyzes customer behavior, compares multiple classification models, selects the best-performing model, and deploys it as a real-time prediction API with an interactive web interface.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Project Workflow](#-project-workflow)
- [Dataset](#-dataset)
- [Data Cleaning](#-1-data-cleaning)
- [Exploratory Data Analysis](#-2-exploratory-data-analysis)
- [Feature Engineering & Encoding](#-3-feature-engineering--encoding)
- [Train Test Split](#-4-train-test-split)
- [Machine Learning Models](#-5-machine-learning-models)
- [Model Evaluation](#-6-model-evaluation)
- [Model Selection](#-7-model-selection)
- [Model Saving](#-8-model-saving)
- [FastAPI Deployment](#-9-fastapi-deployment)
- [API Endpoints](#-10-api-endpoints)
- [Prediction Pipeline](#-11-prediction-pipeline)
- [Risk Classification](#-12-risk-classification)
- [Web Interface](#-13-web-interface)
- [Prediction Result](#-14-prediction-result)
- [Project Structure](#-15-project-structure)
- [Technologies](#-16-technologies-used)
- [How to Run](#-17-how-to-run-the-project)
- [API Documentation](#-18-api-documentation)
- [Screenshots](#-19-project-screenshots)
- [Business Value](#-20-business-value)
- [Limitations](#-21-project-limitations)
- [Future Improvements](#-22-future-improvements)
- [What I Learned](#-23-what-i-learned)
- [Author](#-author)

---

# 📌 Project Overview

Customer churn is one of the most important challenges for companies that provide subscription-based services.

When customers leave a company, the business can lose current revenue as well as future revenue.

The goal of this project is to build a Machine Learning system capable of identifying customers who are more likely to churn.

The project takes customer information such as:

- Demographic information
- Customer tenure
- Internet services
- Security services
- Streaming services
- Contract type
- Payment method
- Monthly charges
- Total charges

and uses this information to predict the probability that a customer will churn.

The project was developed as a complete workflow rather than stopping at model training.

```text
Dataset
   ↓
Data Cleaning
   ↓
Exploratory Data Analysis
   ↓
Feature Preparation
   ↓
Encoding
   ↓
Train/Test Split
   ↓
Multiple ML Models
   ↓
Model Evaluation
   ↓
Best Model Selection
   ↓
Model Saving
   ↓
FastAPI Deployment
   ↓
Input Validation
   ↓
Prediction
   ↓
Risk Classification
   ↓
HTML Web Interface
