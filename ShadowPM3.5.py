############ Shadow PM 3.0 ############



######### Importing Libraries #########

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit

from scipy.stats import randint as sp_randint
from scipy.stats import uniform as sp_uniform

import lightgbm as lgb
import openpyxl as pxl

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

######### Setting Global Parameters #########

# Path to home directory where are all files are created and stored
directory = 'All_Data/'

# Year for which prediction testing is done. 
# Hence if we want to get results for 2022, this should be set as 2021.
# As data from 2021 will be used to predict for 2022.
final_pred_yr = 2023

# This contains year for which data is known and will be used for training the model.
# For example, if we have data till 2021 then 1 Year Forward Returns will be till 2020.
# Max value should be less than 2021 then.
predictionYears = [x for x in range(2013,final_pred_yr)]

# Stocks to drop
stocksToDrop = []





######### Defining Functions ########

"""
Computes a metric a called Directional Accuracy.
Element-wise multiples Y_train and Y_pred.
Returns percentage of elements greater than 0.
"""

def directionalAccuracy(Y_train, Y_pred):
    return np.mean(Y_train*Y_pred >= 0)*100

"""
Returns results on df_predictionYear using the trained model as a dataframe in df_pred.
Calculates metric same as estimateModelPerformance function.
Also adds testing results to df_analysis.
"""
def predictingReturns(model, scaler, df_predictionYear, year):
    global df_analysis
    # Removing columns not used for forward propagation (same was done for df_trainingYears)
    X_sector = df_predictionYear['SECTOR'].tolist()
    X = df_predictionYear.drop(columns = ['1Y_FORWARD_RETURNS', 'DATE', 'NEXT_PX', 'STOCK_NAME', 'SECTOR'], inplace = False)
    cols = X.columns
    X = scaler.transform(X)
    X = pd.DataFrame(X, columns = cols)
    X['SECTOR'] = X_sector
    stocknames, dates, Y = df_predictionYear['STOCK_NAME'], df_predictionYear['DATE'], df_predictionYear['1Y_FORWARD_RETURNS']
    Y_pred = model.predict(X, num_iteration = model.best_iteration_)
    df_pred = pd.DataFrame({'DATE': dates,
                            'STOCK_NAME': stocknames,
                            'NEXT_PX': Y,
                            'NEXT_PX_PRED': Y_pred
                            })
    directional_accuracy = directionalAccuracy(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    correlation = np.corrcoef(Y_pred, Y)[0,1]
    # print(f"Mean Squared Error: {mse}")
    # print(f"Directional Accuracy: {directional_accuracy}")
    # print(f"R^2: {r2}")
    # print(f"Correlation: {correlation}")
    # Storing results for df_predictionYears in df_analysis
    df_analysis.loc[0, 'MSE_Test'] = mse
    df_analysis.loc[0, 'Directional_Accuracy_Test'] = directional_accuracy
    df_analysis.loc[0, 'R^2_Test'] = r2
    df_analysis.loc[0, 'Test Correlation'] = correlation
    return df_pred
"""
Used to calculate model performance using 4 metrics:
1. Directional Accuracy
2. R2 score
3. Mean Squared Error
4. Correlation

It stores these values in df_analysis
"""
def estimateModelPerformance(model, X, Y):
    global df_analysis
    print("\nStoring performance of the Model on Training set....\n")
    Y_pred = model.predict(X, num_iteration = model.best_iteration_)
    directional_accuracy = directionalAccuracy(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    correlation = np.corrcoef(Y_pred, Y)[0,1]
    df_analysis.loc[0, 'R^2_Train'] = r2
    df_analysis.loc[0, 'MSE_Train'] = mse
    df_analysis.loc[0, 'DirAcc_Train']=directional_accuracy
    df_analysis.loc[0, 'corr_Train'] =correlation
def estimateModelPerformanceVal(model, X, Y):
    global df_analysis
    print("\nStoring performance of the Model on Validation set .....\n")
    Y_pred = model.predict(X, num_iteration = model.best_iteration_)
    directional_accuracy = directionalAccuracy(Y, Y_pred)
    r2 = r2_score(Y, Y_pred)
    mse = mean_squared_error(Y, Y_pred)
    correlation = np.corrcoef(Y_pred, Y)[0,1]
    df_analysis.loc[0, 'R^2_Val'] = r2
    df_analysis.loc[0, 'MSE_Val'] = mse
    df_analysis.loc[0, 'DirAcc_Val']=directional_accuracy
    df_analysis.loc[0, 'corr_Val'] =correlation

'''
Takes SPX_Prices.xlsx as input which is imported as allPrices in a DataFrame.
Filters the allPrices DataFrame for the given year. Then iterates over all the stocks. 
Drops a stock if it has same last 30 price values.
Returns an array containing names of stocks to drop.  
'''

def drop_stock(year):

    yearlyPrice = allPrices.copy()
    yearlyPrice = yearlyPrice[yearlyPrice['DATE'].dt.year == year].reset_index(drop = True)
    # First column is date so skipping. Rest 200 are stocks.
    for col in yearlyPrice.columns[1:]:

        data = yearlyPrice[col].tolist()
        data = data[-30::]

        if data.count(data[-1]) == len(data):
          stocksToDrop.append(col)
          print(f'Dropping stock --> {col}, {year}')

    return stocksToDrop


'''
Let us assume, we need predictions for 2024, which means this model will make predictions on 01/01/2023. 
Therefore, finak_pred_yr = 2023.
Now the following split is done for backtesting and training for preceeding years.
Now we split the data with a 13 month difference between last training data and prediction date
i.e. if you are on 01/01/2021, you need predictions for 2022, you train the model only till as 01/12/2019 as we
have 1Y_forward_returns for data till 01/12/19 (i.e. stock prices on 01/12/20)
'''

def yearWiseData(df, year, stocksToDrop):
    
    # Takes 1st month data next to the prediction year 
    df_predictionYear = df[df['DATE'].dt.year == year]
    df_predictionYear = df_predictionYear[df_predictionYear['DATE'].dt.month == 1]
    df_predictionYear = df_predictionYear[~df_predictionYear['STOCK_NAME'].isin(stocksToDrop)]
    # Takes all data before prediction year-1
    #Maintains a difference of 13 months between the last training point and tetsing date
    df_trainingYears = df[df['DATE'].dt.year < year-1]
    df_trainingYears =  df_trainingYears[~df_trainingYears['STOCK_NAME'].isin(stocksToDrop)]
    return df_trainingYears, df_predictionYear

'''
Finding best parameters for LightGBM model.
'''

def optimizeModel(X_train, Y_train, X_val, Y_val, predictionYear, X_train_weight):
     global scorePerYear
     # LGBM Modelling: RANDOMISED GRIDSEARCH . Training LGBM Regressor with Optimal Params
     param_test = {'num_leaves': sp_randint(6, 4000),
                   'min_child_samples': sp_randint(10, 400),
                   'min_child_weight': [1e-1,5e-1,5e-2, 1, 1e1, 1e2],
                   'subsample': sp_uniform(loc=0.3, scale=0.7),
                   'colsample_bytree': sp_uniform(loc=0.4, scale=0.6),
                   'reg_alpha': [1e-6,1e-5,1e-4,1e-3, 5e-3, 1e-2, 5e-2, 2e-1],
                   'reg_lambda':[1e-6,1e-5,1e-4,1e-3, 5e-3, 1e-2, 5e-2, 2e-1],
                   'learning_rate' :[1e-6,1e-5,1e-4,1e-3, 5e-3, 1e-2, 5e-2, 2e-1],
                   'max_depth' : [3,4,5,6,7,8,9,10,11,12],
                   }

     fit_params = {#"early_stopping_rounds" :30,
                   'eval_set' : [(X_val,Y_val)],
                   'eval_names': ['valid'],
                   'verbose': 5000,
                   'categorical_feature': ['SECTOR'],
                   'sample_weight' : X_train_weight
                   }
    
     tscv = TimeSeriesSplit(n_splits=5, test_size=int(len(X_train)/6) , gap=1)

     clf = lgb.LGBMRegressor(max_depth=-1, random_state=314, silent=True, metric='mse', 
                             n_jobs= -1, n_estimators = 1500,boosting_type='dart')
     
     gs = RandomizedSearchCV(estimator=clf, param_distributions=param_test, n_iter=15, cv=tscv, 
                             random_state=314, verbose=True, refit = True, return_train_score=True)
     
     gs.fit(X_train, Y_train, **fit_params)
     scorePerYear.append([gs.best_score_ , gs.cv_results_])
     opt_params = gs.best_params_    

     return opt_params, fit_params





"""
Takes in predictions made on df_predictionYear.
Calculates F1 score based on actual top/bottom stocks and predicted top/bottom stocks
"""

def topBottomAnalyis(df_pred, ascending):
    print(f"Conducting topBottomAnalysis when ascending is {ascending}")
    global df_analysis

    df = df_pred.copy()
    df = df[df['DATE'].dt.month == 1]

    # Ascending False --> Highest at top
    # Ascending True --> Lowest at top
        
    df_px_pred = df.copy()
    df_px = df.copy()
    #Sorting dataframe on the basis of prediction
    df_px_pred = df_px_pred.sort_values(by='NEXT_PX_PRED', ascending=ascending) # changed
    #Sorting dataframe on the basis of actual next price
    df_px = df_px.sort_values(by='NEXT_PX', ascending=ascending)

    stocks_px_pred = df_px_pred['STOCK_NAME'].tolist()
    stocks_px = df_px['STOCK_NAME'].tolist()

    tp = 0  # True Positive
    fp = 0  # False Positive
    fn = 0  # False Negative

    wrong_stocks_30 = 0
    wrong_stocks_50 = 0

    # Assume ascending --> False (Calculating mertic for top stocks)

    # Means if top30 predicted are within 50 of actual, then tp else fp
    for i in range(30):
        if stocks_px.index(stocks_px_pred[i]) < 50:
            tp += 1 
        else:
            fp += 1 

    # Counts how many actual top30 are outside predicted top50
    for i in range(30):
        if stocks_px_pred.index(stocks_px[i]) > 50:
            fn = fn + 1

    # Checks if predicted top 30 are outside top 100 of actual
    for i in range(30):
        if stocks_px.index(stocks_px_pred[i]) > 100:
            wrong_stocks_30 += 1 

    # Checks if predicted top 50 are outside top 100 of actual
    for i in range(50):
        if stocks_px.index(stocks_px_pred[i]) > 100:
            wrong_stocks_50 += 1 

    precision = tp/(tp + fp) # changed
    recall = tp/(tp + fn)
        
    f1 = (2*recall*precision)/(recall+precision)


    # Checking predictions for top30 and top50
    if ascending == False:
        
        # print(f'F1_score_top: {f1}')    
        # print(f'Precision_top: {precision}')
        # print(f'Recall_top: {recall}')

        df_analysis.loc[0, 'F1_score_top'] = f1
        df_analysis.loc[0, 'Precision_top'] = precision
        df_analysis.loc[0, 'Recall_top'] = recall

        df_analysis.loc[0, 'wrong_stocks_top_30'] = wrong_stocks_30
        df_analysis.loc[0, 'wrong_stocks_top_50'] = wrong_stocks_50

    if ascending == True:
        
        # print(f'F1_score_bottom: {f1}')    
        # print(f'Precision_bottom: {precision}')
        # print(f'Recall_bottom: {recall}')

        df_analysis.loc[0, 'F1_score_bottom'] = f1
        df_analysis.loc[0, 'Precision_bottom'] = precision
        df_analysis.loc[0, 'Recall_bottom'] = recall

        df_analysis.loc[0, 'wrong_stocks_bottom_30'] = wrong_stocks_30
        df_analysis.loc[0, 'wrong_stocks_bottom_50'] = wrong_stocks_50





"""
This function calculates a lot of metrics. It updates the df_analysis,  df_combine_top
and df_combine_bottom dataframe. Stores all these results in 'nav.xlsx'.

Metrics calculated:
1. Within +-50%
2. Rank Error

3.1 Common top
3.2 Common bottom

4.1 NAV Top Actual
4.2 Nav Top Predicted
4.3 NAV Bottom Actual
4.4 Nav Bottom Predicted
"""

def nav(df_pred, ascending, sheetname, year):
  global final_pred_yr
  global count_top
  global count_bottom
  global df_combine_top
  global df_combine_bottom
  global df_analysis
  print("Preparing NAV File.....")
  predicted = 0
  count_50 = 0
  count_rank = 0
  df = df_pred.copy()
  df = df[df['DATE'].dt.month == 1]
  val1 = df['NEXT_PX_PRED'].tolist()
  val2 = df['NEXT_PX'].tolist()
  for i in range(len(val1)):
    val = abs(val1[i] - val2[i])
    if val < (abs(val2[i])/2):
      count_50 += 1
  if ascending == False:
    result = count_50/len(val1)
    df_analysis.loc[0, 'Within_+-50%'] = result
    
    # print(f'\nWithin +-50% of actual return: {result}')

  # Sorting based on actual 'px' and 'predicted px'
  df_px_pred = df.copy()
  df_px = df.copy()

  df_px_pred = df_px_pred.sort_values(by='NEXT_PX_PRED', ascending=ascending)
  df_px = df_px.sort_values(by='NEXT_PX', ascending=ascending)
  stocks_px_pred = df_px_pred['STOCK_NAME'].tolist()
  stocks_px = df_px['STOCK_NAME'].tolist()
  
  # Calculating 'rank error'
  
  for i in range(len(stocks_px_pred)):
    count_rank += abs(i - stocks_px.index(stocks_px_pred[i]))
    
  if ascending == False:
    result = count_rank/len(stocks_px_pred)
    df_analysis.loc[0, 'rank_error'] = result

    # print(f'Avg_rank_difference: {result}')
  # Keeping first 30 stocks from both dataframes
  #Metric invalid for final_pred_year
  df_px_pred = df_px_pred.iloc[:30,:]
  df_px = df_px.iloc[:30,:]
  
  stocks_px_pred = df_px_pred['STOCK_NAME'].tolist()
  stocks_px = df_px['STOCK_NAME'].tolist()
  
  # Using allPrices DataFrame (SPX_Prices.xlsx)
  
  dates = allPrices['DATE'].tolist()
  allPrices_px_pred = pd.DataFrame({'DATE' : dates})
  allPrices_px = pd.DataFrame({'DATE' : dates})
  # Just making 2 new allPrices dataframes with ordering as per 'df_px_pred' and 'df_px' with 30 stocks only
  
  for i in range(len(stocks_px_pred)):
    allPrices_px_pred[stocks_px_pred[i]] = allPrices[stocks_px_pred[i]].tolist()
    
  for i in range(len(stocks_px)):
    allPrices_px[stocks_px[i]] = allPrices[stocks_px[i]].tolist()
    
  # Getting unique stock names with ordering as per 'df_px_pred' and 'df_px'. Converting to sets to find intersection.
  
  stocks_px_pred = set(stocks_px_pred) 
  stocks_px = set(stocks_px) 
  
  if ascending == False:
    common_top = len(stocks_px_pred.intersection(stocks_px))
    df_analysis.loc[0, 'Common_top'] = common_top
    
    # print(f'Common_Stocks_Top : {common_top}')
    
  if ascending == True:
    common_bottom = len(stocks_px_pred.intersection(stocks_px))
    df_analysis.loc[0, 'Common_bottom'] = common_bottom
    
    # print(f'Common_Stocks_Bottom : {common_bottom}')


  yearPrice_px_pred = allPrices_px_pred[allPrices_px_pred['DATE'].dt.year == year]

  df_combine_data = pd.DataFrame({'DATE' : dates})
  df_combine_data = df_combine_data[df_combine_data['DATE'].dt.year == year]

  df_val1 = yearPrice_px_pred.copy()
  df_val2 = yearPrice_px_pred.copy()

  initial = np.array([100/30 for i in range(30)])
  df_val2.iloc[0,1:] = initial
  for i in range(1,len(yearPrice_px_pred)):
    df_val1.iloc[i,1:] = (yearPrice_px_pred.iloc[i,1:] - yearPrice_px_pred.iloc[i-1,1:])/(yearPrice_px_pred.iloc[i-1,1:])
    df_val2.iloc[i,1:] = df_val2.iloc[i-1,1:] + df_val2.iloc[i-1,1:]*df_val1.iloc[i,1:]
    
  df_val2['sum'] = df_val2.iloc[:,1:].sum(axis = 1)
  predicted = df_val2.iloc[-1,-1]

  if ascending == False:
    df_combine_data['sum'] = df_val2.iloc[:,1:-1].sum(axis = 1)*(count_top/100)
    df_combine_top = pd.concat([df_combine_top, df_combine_data])
    count_top = df_combine_top.iloc[-1,-1]

  if ascending == True:
    df_combine_data['sum'] = df_val2.iloc[:,1:-1].sum(axis = 1)*(count_bottom/100)
    df_combine_bottom = pd.concat([df_combine_bottom, df_combine_data])
    count_bottom = df_combine_bottom.iloc[-1,-1]

  excel_book = pxl.load_workbook(os.path.join(directory, f"NAV3.5.xlsx"))
  
  with pd.ExcelWriter(os.path.join(directory, f"NAV3.5.xlsx"), engine='openpyxl') as writer:
    
    writer.book = excel_book
    writer.sheets = {worksheet.title: worksheet for worksheet in excel_book.worksheets}

    df_val2.to_excel(writer, sheet_name = sheetname, index=False)
    df_combine_top.to_excel(writer, sheet_name = 'combine_top', index=False)
    df_combine_bottom.to_excel(writer, sheet_name = 'combine_bottom', index=False)

    writer.save()
     
  yearPrice_px = allPrices_px[allPrices_px['DATE'].dt.year == year]
  df_combine_data = df_combine_data[df_combine_data['DATE'].dt.year == year]

  df_val1 = yearPrice_px.copy()
  df_val2 = yearPrice_px.copy()

  df_val2.iloc[0,1:] = (yearPrice_px.iloc[1,1:]*100)/(yearPrice_px.iloc[1,1:]*30)
  
  for i in range(1,len(yearPrice_px)):
    df_val1.iloc[i,1:] = (yearPrice_px.iloc[i,1:] - yearPrice_px.iloc[i-1,1:])/(yearPrice_px.iloc[i-1,1:])
    df_val2.iloc[i,1:] = df_val2.iloc[i-1,1:] + df_val2.iloc[i-1,1:]*df_val1.iloc[i,1:] 

  df_val2['sum'] = df_val2.iloc[:,1:].sum(axis = 1)

  actual = df_val2.iloc[-1,-1]
        
  if ascending == False:
    # print(f'actual_return_top: {actual}')
    # print(f'predicted_return_top: {predicted}') 

    df_analysis.loc[0, 'NAV_Top_Actual'] = actual
    df_analysis.loc[0, 'NAV_Top_Predicted'] = predicted

  if ascending == True:

    # print(f'actual_return_bottom: {actual}')
    # print(f'predicted_return_bottom: {predicted}') 
    
    df_analysis.loc[0, 'NAV_Bottom_Actual'] = actual
    df_analysis.loc[0, 'NAV_Bottom_Predicted'] = predicted





######### Importing The Dataset #########

## Reading 12 Month and 0 Month lag training data generated by `DataSegregation_12M.py` and `DataSegregation_0M.py` ##

# LAG means file has all columns similar but 1Y_Forward_Returns column has been calculated from 1 year future prices.
# So 12month lag has all columns similar as 0month lag but 1Y_Forward_Returns has values calculated from next year. This column is also the target value.

file_12m = '_Data.csv' 
file_0m = '_Data1.csv' 
print("Opening Workbooks ....")
df_12m = pd.read_csv(os.path.join(directory,file_12m))
df_0m = pd.read_csv(os.path.join(directory,file_0m))
print("Workbooks Opened")



## Formating Data ##

# Making column names upper case and replacing spacings with '_'
df_12m.columns = [str(x).replace(' ', '_').upper() for x in df_12m.columns]
df_0m.columns = [str(x).replace(' ', '_').upper() for x in df_0m.columns]

# Setting Date-Time column
df_12m['DATE'] = pd.to_datetime(df_12m['DATE'])
df_0m['DATE'] = pd.to_datetime(df_0m['DATE'])

# Filling NA values with 0
df_12m = df_12m.fillna(0)
df_0m = df_0m.fillna(0)

# Label Encoding SECTOR column
le = LabelEncoder()
# print(df_12m['SECTOR'])
le.fit(df_12m['SECTOR'].astype(str))

df_12m['SECTOR'] = le.transform(df_12m['SECTOR'].astype(str))
df_0m['SECTOR'] = le.transform(df_0m['SECTOR'].astype(str))

sector_dict = dict(zip(le.classes_, le.transform(le.classes_)))
# print('Sector Mapping')
# print(sector_dict)
# print("BT Point 1 cleared")

# Printing Size of Data
#print(f'Length of 12M data --> {len(df_12m)}')
#print(f'Length of 0M data --> {len(df_0m)}')



## Setting Train and Prediction Data ##

# Training data which will be also used for train and val split
# Contains data till final_pred_year-1 but due to 12 month lags effectively has forward returns till final_pred_year
#If file contains Data from final_pred_yr -1 to final_pred_yr with one year forward returns, change the below line
df_model = df_12m[df_12m['DATE'].dt.year < final_pred_yr]


# Prediction data
# Contains data corresponding to final_pred_year since there is 0 month lag
df_predict = df_0m[df_0m['DATE'].dt.year == final_pred_yr]
df_predict = df_predict[df_predict['DATE'].dt.month==1]
print(f'Length of 12M data --> {len(df_model)}')
print(f'Length of 0M data --> {len(df_predict)}')




## Loading Prices DataFrame ##

# Contains Prices for all 200 equities with first column as Date (Total 201 columns)
allPrices = pd.read_excel(os.path.join(directory, 'SPX_Prices.xlsx'), engine = 'openpyxl')
allPrices.rename(columns = {'Date':'DATE'}, inplace = True)

## Dataframes to store analysis results ##

analysis_cols = ['Year', 'NAV_Top_Actual', 'NAV_Top_Predicted', 'Common_top', 'F1_score_top', 
                 'Precision_top', 'Recall_top', 'MAE', 'Directional_Accuracy_Test', 'R^2_Test', 
                 'R^2_train', 'Within_+-50%', 'rank_error', 'wrong_stocks_top_30', 'wrong_stocks_top_50', 
                 'NAV_Bottom_Actual', 'NAV_Bottom_Predicted', 'Common_bottom', 'F1_score_bottom', 
                 'Precision_bottom', 'Recall_bottom', 'wrong_stocks_bottom_30', 'wrong_stocks_bottom_50']

# Used for storing results while calculating and saved as analysis
df_analysis = pd.DataFrame(columns  = analysis_cols)
df_analysis_final = pd.DataFrame(columns  = analysis_cols)

# print("BT Point 4 cleared")
# Used in nav function
wb = pxl.Workbook()
wb.save(filename=os.path.join(directory, f"NAV3.5.xlsx"))

df_combine_top = pd.DataFrame(columns = ['DATE', 'sum'])
df_combine_bottom = pd.DataFrame(columns = ['DATE', 'sum'])
# print("BT Point 5 cleared")
count_top = 100
count_bottom = 100




######### Training and the Model and Predicting Returns #########

# Writing Dataframe to Excel Sheets using pd.ExcelWriter

with pd.ExcelWriter(os.path.join(directory, f'Returns3.5.xlsx'), engine='openpyxl') as writer:

    for year in predictionYears:
        #[2019, 2020, 2021, 2022]

        drop_stock(year)

        print(f"--------- Currently working on predictions of {year+1} i.e. training till 01/12/{year-1} and testing on 01/01/{year+1} -------- ")
        print('\nDEFINING INPUT AND OUTPUT\n')
        if year+1 == final_pred_yr: 
          df_trainingYears, df_predictionYear = df_model, df_predict  
        else:   
          df_trainingYears, df_predictionYear = yearWiseData(df_model, year+1, stocksToDrop)
        print(f"\nTraining Set Entries: {len(df_trainingYears)}, Test Year Entries: {len(df_predictionYear)}\n")
        traindata = df_trainingYears.copy()
        testdata =df_predictionYear.copy()
        # traindata.to_excel(f"TrainingData3.5_SPX_{final_pred_yr}.xlsx")
        # testdata.to_excel(f"Testdata3.5_SPX_{final_pred_yr}.xlsx") 
        print('\nScaling Data\n')
        X = df_trainingYears
        X_date = X['DATE'].tolist()
        X_returns = X['1Y_FORWARD_RETURNS'].tolist()
        X_stocks = X['STOCK_NAME'].tolist()
        X_nextpx = X['NEXT_PX'].tolist()
        X_sector = X['SECTOR'].tolist()

        X = X.drop(columns = ['NEXT_PX','1Y_FORWARD_RETURNS','STOCK_NAME', 'DATE', 'SECTOR'], inplace = False)
        cols = X.columns
        scaler = MinMaxScaler(feature_range = (0, 1))
        X = scaler.fit_transform(X)
        X = pd.DataFrame(X, columns = cols) 
        
        X['DATE'] = X_date
        X['1Y_FORWARD_RETURNS'] = X_returns
        X['STOCK_NAME'] = X_stocks
        X['NEXT_PX'] = X_nextpx
        X['SECTOR'] = X_sector
        print('\nPREPROCESSING & SPLITTING THE DATASET\n')

        # Splitting Data into train and validation sets
        X_t, X_v = train_test_split(X, test_size = 0.2, random_state = 0, shuffle = False)
        X_t = X_t.sort_values(by = 'DATE') 

        X_train = X_t.drop(columns = ['NEXT_PX','1Y_FORWARD_RETURNS', 'STOCK_NAME', 'DATE'], inplace = False)
        Y_train = X_t['1Y_FORWARD_RETURNS']

        X_val = X_v.drop(columns = ['NEXT_PX','1Y_FORWARD_RETURNS', 'STOCK_NAME', 'DATE'], inplace = False)
        Y_val = X_v['1Y_FORWARD_RETURNS']

        # Setting weights for each row
        X_train_weight = np.zeros(len(X_t))
        # print("BT point 8 cleared")

        count_weight = 1
        
        for i in range(len(X_train_weight)):
          X_train_weight[i] = count_weight 
          if(i%2500 == 0):
            count_weight += 1 
                
        X_train_weight = X_train_weight/count_weight

        cols = X_train.columns
        scorePerYear = []
        # print("BT point 9 cleared")

        if year != final_pred_yr:
            print(f'Training for {year}')
            print("\nLight Gradient Boosting Algorithm Training:\n")

            """ 
            Note: n_estimators is set to a "large value". The actual number of trees build will 
            depend on early stopping and 5000 define only the absolute maximum
            """

            opt_params, fit_params = optimizeModel(X_train, Y_train, X_val, Y_val, year, X_train_weight)

            model = lgb.LGBMRegressor(max_depth=-1,random_state=314, silent=True, metric='mse', 
                                      n_jobs=-1, n_estimators = 1500,boosting_type='dart')
            model.set_params(**opt_params)
    
            model.fit(X_train, Y_train, eval_set = [(X_val, Y_val), (X_train, Y_train)], 
                      eval_metric = 'l1', eval_names = ['valid','train'], verbose= 100000, 
                      categorical_feature = fit_params['categorical_feature'],
                      sample_weight = fit_params['sample_weight'])
            
            #lgb.plot_metric(model)
    
            feature_importance = pd.DataFrame({
                'feature' : cols ,
                'importance_freq' : model.feature_importances_,
                'importance_gain' : model._Booster.feature_importance(importance_type='gain')
                })
            
            feature_importance.to_excel(writer, sheet_name=(f'{year}_features'), index=False)
            
            # Estimating Performance of the Model on Training/Validation

            print('\nEstimating model Performance on Training Data\n')
            estimateModelPerformance(model, X_train, Y_train)
            print('\nEstimating model Performance on Validation Data\n')
            estimateModelPerformanceVal(model, X_val, Y_val)

        # Predicting returns on df_Predictionyear using trained model.

        print(f"\nPredicting Returns On Year: {year+1}\n")
        df_pred = predictingReturns(model, scaler, df_predictionYear, year)
        # df_pred = df_pred[df_pred['DATE'].dt.year==year+1]
        df_pred.to_excel(writer, f"{year}_returns", index=False)

        df_analysis.loc[0, 'Year'] = year + 1

        # Calculating F1 scores for top and bottom stock predictions
        topBottomAnalyis(df_pred, ascending = False)
        topBottomAnalyis(df_pred, ascending = True)
        # NAV file
        nav(df_pred = df_pred, ascending = False, sheetname = f"{year+1}_top", year = year+1)
        nav(df_pred = df_pred, ascending = True, sheetname = f"{year+1}_bottom", year = year+1)

        df_analysis_final = pd.concat([df_analysis_final, df_analysis])
        df_analysis_final.to_excel(writer, sheet_name='analysis', index=False)
        
        # print(f"score per year is {scorePerYear}")
      
  
      

################################################################################################################################################


