# Shadow PM 3.5

Shadow PM 3.5 is a predictive modeling project for stock market analysis using machine learning. It leverages historical stock data to predict future stock price movements and analyze the performance of a trading strategy.

For a detailed tutorial on the code and its usage, please refer to [this link](https://drive.google.com/file/d/1IQnMbn8E1zhCy5qZLgkZP6m4wMMTk7ch/view?usp=share_link).

## Importing Libraries

This project utilizes various Python libraries and packages, including:

- `os`: For file and directory operations.
- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computations.
- `matplotlib`: For data visualization.
- `scikit-learn`: For machine learning algorithms and metrics.
- `lightgbm`: A gradient boosting framework.
- `openpyxl`: For Excel file handling.
- `warnings`: To suppress certain warning messages.

## Setting Global Parameters

The project defines several global parameters:

- `directory`: The path to the home directory where data files are created and stored.
- `final_pred_yr`: The year for which prediction testing is performed. Data up to this year is used for training.
- `predictionYears`: A list of years for which data is known and will be used for training.
- `stocksToDrop`: A list of stock names to be excluded from the analysis.

## Defining Functions

The project includes various functions for tasks such as calculating directional accuracy, estimating model performance, optimizing model parameters, analyzing top and bottom stocks, and computing NAV (Net Asset Value).

## Importing The Dataset

The project reads and processes two datasets: one with a 12-month lag and another with a 0-month lag. These datasets contain stock data used for training and testing the predictive model.

## Training and the Model and Predicting Returns

The code trains a predictive model for stock price movements and makes predictions for the desired years. It includes preprocessing, splitting the dataset into training and validation sets, optimizing model parameters, and evaluating the model's performance. Predictions are made for each specified year, and various metrics are computed for analysis.

## Data Analysis and Visualization

The project includes data analysis and visualization steps to assess the performance of the predictive model. Metrics such as Mean Squared Error (MSE), Directional Accuracy, R-squared (R^2), correlation, and more are calculated and stored in an Excel file for further analysis.
Analysis sheet can be found in returns excel file which gives overall overview of Model's performance.

## Net Asset Value (NAV) Calculation

The project calculates the Net Asset Value (NAV) for both top and bottom stocks based on the predicted and actual returns. The results are stored in an Excel file for analysis.

### What is NAV?

If we allocate the resources(e.g - 100 dollars) equally among top n stocks the cummulative return of every stock we get after a year is defined as net asset value.

## Conclusion

Shadow PM 3.5 is a comprehensive project for stock market analysis and prediction. It uses machine learning techniques and various metrics to assess the performance of the predictive model. The results and analysis are stored in Excel files for further evaluation.
