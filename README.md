
# Lighthouse Labs midterm project

## Outline of steps:

1. Connect to Postgres database via JDBC (SQL Workbench)
    - Find train and test flights data
    - Randomly extract 10000 rows of flights train data (without NaN values in identifying columns)

2. Save data to .txt file and use Pandas in Python to display dataframe
    - exploratory_analysis.ipynb

3. Decide which features to keep in initial attempt
    - Drop columns containing high percentage of NaN values
    - Drop columns containing identifying columns (i.e. flight ID. location ID)
    - Check counts in each column - if value counts are either too high or too low, drop column

4. Pre-process data for initial attempt
    - Drop columns containing non-numerical data (will later return to these columns and perform one-hot encoding)
    - Use scaling fit_transform from sklearn on remaining numerical features
    - Split training data into sub-training and sub-testing data

5. Perform an initial regression attempt with the reduced features and sub-training data
    - modeling.ipynb
    - Check that R-squared value is > 0 (better than random)

6. Increase model complexity using other regression methods
    - Polynomial
    - Logistic
    - Ridge
    - Lasso
    - SVR
    - Random forest
    - Gradient boosting
    - XGBoost

7. Choose best model based on regression metrics
    - R-squared
    - Root mean squared error

8. Integrate weather API based on location
    - Departure
    - Arrival

9. Perform GridSearchCV() method on best model
    - Automatically perform k-fold cross-validation

10. Time-permitting only:
    - Stretch material (classification problems)
    - Historical data as predictors
        - Fuel consumption
        - Passengers
    
11. Perform predictions on flights test data
    - Use selected model with optimized parameters via grid search

12. Transfer code blocks to .py files and submit final CSV file