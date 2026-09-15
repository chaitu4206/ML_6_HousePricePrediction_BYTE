# Task 6 Summary: House Price Prediction Using Linear Regression

## Objective

The objective of this task was to build a Linear Regression model to predict house prices using the Kaggle House Prices dataset.

## Dataset

The dataset contained 1460 records and 81 columns.

The target variable was:

- `SalePrice`

The following features were selected for model training:

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

## Methodology

1. Loaded the dataset using Pandas.
2. Selected relevant numerical features.
3. Separated input features and target variable.
4. Split the data into training and testing sets.
5. Used median imputation to handle missing values.
6. Applied StandardScaler for feature scaling.
7. Trained a Linear Regression model.
8. Evaluated the model using RMSE, MAE, and R² Score.
9. Saved the trained model using Joblib.
10. Generated actual-versus-predicted and residual plots.

## Train-Test Split

- Training samples: 1168
- Testing samples: 292
- Test size: 20%
- Random state: 42

## Results

| Evaluation Metric | Value |
|---|---:|
| RMSE | $38,834.50 |
| MAE | $24,767.43 |
| R² Score | 0.8034 |

## Interpretation

The R² score of 0.8034 means that the model explains approximately 80.34% of the variation in house prices in the test dataset.

The MAE of $24,767.43 means that the model's predictions differ from the actual house prices by approximately $24,767 on average.

The RMSE of $38,834.50 indicates the overall prediction error, with larger errors receiving more weight.

## Generated Files

- `models/linear_regression_house_price.pkl`
- `outputs/evaluation_metrics.csv`
- `outputs/sample_predictions.csv`
- `plots/actual_vs_predicted.png`
- `plots/residual_plot.png`

## Conclusion

A Linear Regression model was successfully developed for house price prediction. The model achieved an R² score of 0.8034, demonstrating a reasonable baseline performance. More advanced regression techniques and feature engineering could improve the results.
