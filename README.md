# Housing Price Prediction — Linear Regression

Task 3 of the Elevate Labs AI & ML internship (Sep–Nov 2025).

## Overview
Linear Regression on the Housing Price Prediction dataset (Kaggle, via kagglehub): preprocessing, train/test split, model fitting, evaluation, and coefficient interpretation.

## Files
- `Task_3.ipynb` — notebook with the full pipeline
- `Housing.csv` — raw input data (545 rows × 13 columns)
- `Housing_Cleaned.csv` — cleaned, encoded, scaled model matrix produced by the notebook

## What it does
1. Download the dataset with `kagglehub` and load it
2. Preprocess with a `ColumnTransformer` (median imputation + `StandardScaler` for numeric columns, most-frequent imputation + one-hot encoding for categorical columns)
3. Train/test split (80/20, `random_state=42`)
4. Fit `LinearRegression` inside a `Pipeline`
5. Evaluate with MAE, MSE, RMSE, and R²
6. Plot predicted vs. actual values and a simple-regression line for the top correlated numeric feature
7. Interpret standardized coefficients

## Results
Exact metrics printed in the notebook (test set):
- MAE: `970,043.4039`
- MSE: `1,754,318,687,330.6689`
- RMSE: `1,324,506.9601`
- R²: `0.6529`

Largest-magnitude coefficients: `bathrooms` (+521,879), `area` (+519,552), `airconditioning_yes` (+395,713), `stories` (+349,251), `hotwaterheating_yes` (+342,325).

## How to run
Open `Task_3.ipynb` in Jupyter and run all cells. The first cell downloads the dataset via `kagglehub`; the notebook then reads/writes CSVs from a hardcoded local cache path — update `INPUT_CSV`/`OUTPUT_CSV` if needed.

Requires: pandas, numpy, seaborn, matplotlib, scikit-learn, kagglehub

## Author
Joshua Jose · MSc Data Science, FAU Erlangen-Nürnberg · linkedin.com/in/j0shuaj0se
