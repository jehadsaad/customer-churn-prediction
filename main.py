from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import xgboost as xgb
import json


# =========================================================
# 1. FastAPI Application
# =========================================================

app = FastAPI(
    title="Customer Churn Prediction API",
    description="API for Customer Churn Prediction using XGBoost",
    version="1.0.0"
)


# =========================================================
# 2. File Paths
# =========================================================

MODEL_PATH = "best_xgb_model.json"
PREPROCESSOR_PATH = "preprocessor.pkl"
FEATURE_INFO_PATH = "feature_info.json"


# =========================================================
# 3. Load XGBoost Model
# =========================================================

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)


# =========================================================
# 4. Load Preprocessor
# =========================================================

preprocessor = joblib.load(PREPROCESSOR_PATH)


# =========================================================
# 5. Load Feature Information
# =========================================================

with open(FEATURE_INFO_PATH, "r") as f:
    feature_info = json.load(f)


# =========================================================
# 6. Binary Features
# =========================================================
#
# These features were converted during model training:
#
# Yes -> 1
# No -> 0
# No internet service -> 0
#
# The API must perform the same transformation.
# =========================================================

BINARY_FEATURES = [
    "Partner",
    "Dependents",
    "PaperlessBilling",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]


# =========================================================
# 7. Allowed Values
# =========================================================

BINARY_VALUES = {
    "Yes",
    "No",
    "No internet service"
}


# =========================================================
# 8. Pydantic Input Model
# =========================================================

class CustomerData(BaseModel):

    SeniorCitizen: int = Field(
        ...,
        ge=0,
        le=1,
        description="Senior citizen: 0 = No, 1 = Yes"
    )

    Partner: str

    Dependents: str

    tenure: int = Field(
        ...,
        ge=0,
        description="Number of months the customer has stayed with the company"
    )

    InternetService: str

    OnlineSecurity: str

    OnlineBackup: str

    DeviceProtection: str

    TechSupport: str

    StreamingTV: str

    StreamingMovies: str

    Contract: str

    PaperlessBilling: str

    PaymentMethod: str

    MonthlyCharges: float = Field(
        ...,
        ge=0
    )

    TotalCharges: float = Field(
        ...,
        ge=0
    )


# =========================================================
# 9. Home Endpoint
# =========================================================

@app.get("/")
def home():

    return {
        "message": "Customer Churn Prediction API",
        "status": "running",
        "model": "XGBoost"
    }


# =========================================================
# 10. Health Check Endpoint
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


# =========================================================
# 11. HTML Frontend
# =========================================================

@app.get("/frontend")
def frontend():

    return FileResponse("index.html")


# =========================================================
# 12. Prediction Endpoint
# =========================================================

@app.post("/predict")
def predict_churn(customer: CustomerData):

    try:

        # -------------------------------------------------
        # Step 1: Convert Pydantic object to dictionary
        # -------------------------------------------------

        customer_dict = customer.model_dump()


        # -------------------------------------------------
        # Step 2: Validate Binary Features
        # -------------------------------------------------

        for column in BINARY_FEATURES:

            value = customer_dict[column]

            if value not in BINARY_VALUES:

                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Invalid value for '{column}'. "
                        f"Allowed values are: "
                        f"{sorted(BINARY_VALUES)}"
                    )
                )


        # -------------------------------------------------
        # Step 3: Convert Dictionary to DataFrame
        # -------------------------------------------------

        input_data = pd.DataFrame(
            [customer_dict]
        )


        # -------------------------------------------------
        # Step 4: Apply Binary Encoding
        # -------------------------------------------------
        #
        # Same preprocessing used during model training:
        #
        # Yes -> 1
        # No -> 0
        # No internet service -> 0
        # -------------------------------------------------

        for column in BINARY_FEATURES:

            input_data[column] = input_data[column].map({
                "Yes": 1,
                "No": 0,
                "No internet service": 0
            })


        # -------------------------------------------------
        # Step 5: Apply Saved Preprocessor
        # -------------------------------------------------

        processed_data = preprocessor.transform(
            input_data
        )


        # -------------------------------------------------
        # Step 6: Generate Churn Probability
        # -------------------------------------------------

        probability = float(
            model.predict_proba(
                processed_data
            )[0][1]
        )


        # -------------------------------------------------
        # Step 7: Make Prediction
        # -------------------------------------------------
        #
        # 0.5 is used as the classification threshold.
        # -------------------------------------------------

        prediction = int(
            probability >= 0.5
        )


        # -------------------------------------------------
        # Step 8: Generate Result
        # -------------------------------------------------

        if prediction == 1:

            result = "Customer is likely to churn"

        else:

            result = "Customer is unlikely to churn"


        # -------------------------------------------------
        # Step 9: Determine Risk Level
        # -------------------------------------------------

        if probability >= 0.70:

            risk_level = "High"

        elif probability >= 0.40:

            risk_level = "Medium"

        else:

            risk_level = "Low"


        # -------------------------------------------------
        # Step 10: Return Prediction
        # -------------------------------------------------

        return {

            "prediction": prediction,

            "churn_probability": round(
                probability,
                4
            ),

            "churn_percentage": (
                f"{probability * 100:.2f}%"
            ),

            "risk_level": risk_level,

            "result": result
        }


    # =====================================================
    # Error Handling
    # =====================================================

    except HTTPException:

        raise


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )