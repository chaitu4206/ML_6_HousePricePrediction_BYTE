# ============================================================
# TASK 6: HOUSE PRICE PREDICTION USING LINEAR REGRESSION
# ============================================================

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Allows the script to work correctly when executed from
# the project root using: python src/train_model.py

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "train.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)


# ============================================================
# 2. LOAD DATASET
# ============================================================

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n"
        "Please place Kaggle train.csv inside the data folder."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 60)
print("HOUSE PRICE PREDICTION USING LINEAR REGRESSION")
print("=" * 60)

print("\nDataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())


# ============================================================
# 3. SELECT FEATURES AND TARGET
# ============================================================

features = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "1stFlrSF",
    "YearBuilt",
    "FullBath",
    "BedroomAbvGr",
    "TotRmsAbvGrd",
    "GarageArea"
]

target = "SalePrice"

# Check whether all required columns exist
required_columns = features + [target]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in dataset: {missing_columns}"
    )

X = df[features]
y = df[target]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])


# ============================================================
# 5. PREPROCESSING + LINEAR REGRESSION PIPELINE
# ============================================================

model = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        ),
        (
            "regressor",
            LinearRegression()
        )
    ]
)


# ============================================================
# 6. TRAIN MODEL
# ============================================================

model.fit(X_train, y_train)

print("\nModel training completed successfully!")


# ============================================================
# 7. PREDICTIONS
# ============================================================

y_pred = model.predict(X_test)


# ============================================================
# 8. EVALUATION METRICS
# ============================================================

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print(f"RMSE     : ${rmse:,.2f}")
print(f"MAE      : ${mae:,.2f}")
print(f"R2 Score : {r2:.4f}")


# ============================================================
# 9. SAVE EVALUATION METRICS
# ============================================================

metrics_df = pd.DataFrame({
    "Metric": ["RMSE", "MAE", "R2 Score"],
    "Value": [rmse, mae, r2]
})

metrics_path = os.path.join(
    OUTPUTS_DIR,
    "evaluation_metrics.csv"
)

metrics_df.to_csv(metrics_path, index=False)


# ============================================================
# 10. RESIDUAL PLOT
# ============================================================

residuals = y_test.values - y_pred

plt.figure(figsize=(10, 6))

plt.scatter(
    y_pred,
    residuals,
    alpha=0.6
)

plt.axhline(
    y=0,
    linestyle="--"
)

plt.xlabel("Predicted House Price")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot - Linear Regression")
plt.grid(True, alpha=0.3)
plt.tight_layout()

residual_plot_path = os.path.join(
    PLOTS_DIR,
    "residual_plot.png"
)

plt.savefig(
    residual_plot_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 11. ACTUAL VS PREDICTED PLOT
# ============================================================

plt.figure(figsize=(10, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

min_value = min(y_test.min(), y_pred.min())
max_value = max(y_test.max(), y_pred.max())

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True, alpha=0.3)
plt.tight_layout()

actual_predicted_path = os.path.join(
    PLOTS_DIR,
    "actual_vs_predicted.png"
)

plt.savefig(
    actual_predicted_path,
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# 12. SAMPLE PREDICTIONS
# ============================================================

sample_predictions = pd.DataFrame({
    "Actual Price": y_test.values,
    "Predicted Price": y_pred
})

sample_predictions["Absolute Error"] = (
    sample_predictions["Actual Price"]
    - sample_predictions["Predicted Price"]
).abs()

# Save all test predictions
predictions_path = os.path.join(
    OUTPUTS_DIR,
    "sample_predictions.csv"
)

sample_predictions.to_csv(
    predictions_path,
    index=False
)

print("\n" + "=" * 60)
print("SAMPLE PREDICTIONS")
print("=" * 60)

print(sample_predictions.head(5).to_string(index=False))


# ============================================================
# 13. SAVE MODEL ARTIFACT
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "linear_regression_house_price.pkl"
)

joblib.dump(model, model_path)


# ============================================================
# 14. FINAL OUTPUT SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("PROJECT COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"Model saved       : {model_path}")
print(f"Metrics saved     : {metrics_path}")
print(f"Residual plot     : {residual_plot_path}")
print(f"Actual-pred plot  : {actual_predicted_path}")
print(f"Predictions saved : {predictions_path}")