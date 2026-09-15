# House Price Prediction Using Linear Regression

## Project Overview

This project predicts house prices using a Linear Regression machine learning model.

The project uses the Kaggle House Prices dataset and applies data preprocessing, feature selection, model training, evaluation, and visualization.

## Dataset

The dataset contains residential house information from Ames, Iowa.

The target variable is:

- `SalePrice`

Selected features include:

- `OverallQual`
- `GrLivArea`
- `GarageCars`
- `TotalBsmtSF`
- `1stFlrSF`
- `YearBuilt`
- `FullBath`
- `BedroomAbvGr`
- `TotRmsAbvGrd`
- `GarageArea`

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Jupyter Notebook
- Joblib

## Machine Learning Workflow

1. Load the dataset
2. Select relevant features
3. Separate features and target
4. Split the dataset into training and testing sets
5. Handle missing values using median imputation
6. Scale numerical features
7. Train the Linear Regression model
8. Evaluate model performance
9. Save the trained model
10. Generate prediction and residual plots

## Model

The model used in this project is:

```text
Linear Regression
```

A preprocessing pipeline was used with:

- SimpleImputer
- StandardScaler
- LinearRegression

## Model Evaluation

The model was evaluated using RMSE, MAE, and R² Score.

| Metric | Result |
|---|---:|
| RMSE | $38,834.50 |
| MAE | $24,767.43 |
| R² Score | 0.8034 |

The R² score of 0.8034 indicates that the model explains approximately 80.34% of the variance in house prices on the test dataset.

## Project Structure

```text
house-price-linear-regression/
├── data/
│   └── train.csv
├── models/
│   └── linear_regression_house_price.pkl
├── notebooks/
│   └── house_price_prediction.ipynb
├── outputs/
│   ├── evaluation_metrics.csv
│   └── sample_predictions.csv
├── plots/
│   ├── actual_vs_predicted.png
│   └── residual_plot.png
├── src/
│   └── train_model.py
├── README.md
├── summary.md
└── requirements.txt
```

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the training script

```bash
python src/train_model.py
```

### 3. View the generated files

After execution, the following files will be generated:

- Trained model in the `models` folder
- Evaluation metrics in the `outputs` folder
- Sample predictions in the `outputs` folder
- Visualizations in the `plots` folder

## Results

The Linear Regression model achieved an R² score of 0.8034, showing reasonable predictive performance for the selected features.

## Future Improvements

- Try Ridge and Lasso Regression
- Apply logarithmic transformation to the target variable
- Perform feature engineering
- Compare different regression algorithms
- Use cross-validation
- Tune model hyperparameters
