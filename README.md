## 🖥️ Web Interface Demo

The project includes an interactive web interface that allows users to enter customer information and receive a real-time churn prediction through the deployed Machine Learning model.

The interface provides a simple way to interact with the prediction API without requiring the user to manually send API requests through Swagger or write JSON input.

### 🔄 How It Works

```text
Customer Information
        ↓
HTML Form
        ↓
FastAPI /predict Endpoint
        ↓
Input Validation
        ↓
Feature Preprocessing
        ↓
XGBoost Model
        ↓
Churn Probability
        ↓
Risk Classification
        ↓
Prediction Result
        ↓
Web Interface
