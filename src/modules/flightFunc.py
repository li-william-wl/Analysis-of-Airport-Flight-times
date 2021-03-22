import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

def txt_to_df(filename, scaler, make_dum = False, to_csv = False, output_name = None):
    '''
    filename (str): the name of the .txt file, including "".txt"
    extension and local folder path
    
    scaler (method): either StandardScaler(), MinMaxScaler(), RobustScaler(),
    or PowerTransformer() from sklearn.preprocessing; the first two methods
    are more sensitive to outliers
    
    make-dum (bool): default False; if True, then dummy variables will be created
    for object-type variables via one-hot encoding
    
    to_csv (bool): default False; if True, then final dataframe will be written
    to a .csv file
    
    output_name (str): the name to append to "x_<output_name>.csv" and
    "y_<output_name>.csv", only required if write = True
    '''
    
    # Import the .txt file (parameter) as a Pandas df
    dfRead = pd.read_csv(filename, delimiter = '\t')
    
    # Separate the numerical feature columns for scaling
    dfNumeric = dfRead[['crs_dep_time', 'crs_arr_time',
                              'crs_elapsed_time', 'distance']]
    
    # Convert the time column into hours
    dfNumeric.loc[:, 'crs_dep_time'] = dfRead.loc[:, 'crs_dep_time'] // 100
    dfNumeric.loc[:, 'crs_arr_time'] = dfRead.loc[:, 'crs_arr_time'] // 100
    
    # Scale the numerical data either with StandardScaler() or
    # MinMaxScaler() from parameter passed in function
    dfNumeric_scaled = scaler.fit_transform(dfNumeric)
    
    # Convert the scaled X Numpy array back into Pandas df with the 
    # original column names and "FT" (fit-transformed)
    colList = dfNumeric.columns
    colList = [col + 'FT' for col in colList]
    dfNumeric_scaled = pd.DataFrame(dfNumeric_scaled, columns = colList)
    
    # Convert numerical ID data into objects for one-hot encoding
    df = dfRead.astype({
    'mkt_carrier_fl_num': object,
    'op_carrier_fl_num': object,
    'origin_airport_id': object,
    'dest_airport_id': object
    })
    
    # Select all of the object-type columns for one-hot encoding
    dfObjects = df.select_dtypes(include = 'object')
    
    # Convert the flight date column to a datetime format
    dfObjects['fl_date'] = pd.to_datetime(dfObjects['fl_date'])
    
    # Concatenate the object-type columns with features dataframe
    dfNumeric_scaled = dfNumeric_scaled.reset_index(drop = True)
    dfObjects = dfObjects.reset_index(drop = True)
    dfConcat = pd.concat([dfNumeric_scaled, dfObjects], axis = 1)
    
    #Drop columns with features highly correlated to features in other
    # columns
    dfConcat.drop(columns = ['mkt_unique_carrier', 'branded_code_share',
                           'mkt_carrier', 'mkt_carrier_fl_num',
                           'origin_airport_id', 'origin_city_name',
                           'dest_airport_id', 'dest_city_name', 'dup',
                            'tail_num', 'op_carrier_fl_num', 'arr_delay',
                            'op_carrier_fl_num','cancellation_code'],
                          inplace = True, errors = 'ignore')
    
    # Use one-hot encoding to create dummy variables for categorical
    # features, designate the correct columns for the target values as well
    # as preserved sign, and drop all rows with NaN values
    if make_dum:
        FinalDF = pd.get_dummies(dfConcat)
    else:
        FinalDF = dfConcat
    FinalDF['arr_delay'] = dfRead['arr_delay']
    FinalDF.dropna(inplace = True)
    X = FinalDF.drop('arr_delay', axis = 1)
    y = FinalDF['arr_delay']

    # Write the feature columns and target columns to "X" and "y" .csv files
    # (parameter "output_name" is appended)
    if to_csv:
        X.to_csv('X_' + output_name + '.csv', index = False, compression = 'gzip')
        y.to_csv('y_' + output_name + '.csv', index = False, compression = 'gzip')
    else:
        return X, y
    
def replaceObjectsWithNums(X, scaler):
    '''
    Replaces in X, columns [carrier, origin, dest] with numerical continuous values for faster modeling
    An alternative to making dummies
    Takes in X, a Dataframe with op_unique_carrier, origin, and dest columns, and a scaler method
    '''
    
    # Average arrival delay time grouped by carrier, as a dictionary:
    
    carrierDict = {
        '9E': 3.788253768330484,
        '9K': -1.4138972809667674,
        'AA': 6.209127910387774,
        'AS': 0.4580709294158264,
        'AX': 15.614108372836077,
        'B6': 11.328905876893792,
        'C5': 23.297226405497323,
        'CP': 5.752920400632577,
        'DL': 0.4649172663471822,
        'EM': 6.439237738206811,
        'EV': 11.460218091834946,
        'F9': 11.294148868243784,
        'G4': 8.948751471435369,
        'G7': 8.645223084384094,
        'HA': 0.7479588660633051,
        'KS': 17.362094951017333,
        'MQ': 6.192537786526977,
        'NK': 5.135043464348568,
        'OH': 7.457289195029419,
        'OO': 7.155529056303303,
        'PT': 4.442924183761644,
        'QX': 2.056287279578388,
        'UA': 7.0310394918378565,
        'VX': 1.7279776132454965,
        'WN': 3.549975537488939,
        'YV': 9.682003383338802,
        'YX': 3.946617621243796,
        'ZW': 7.347722443129507
    }
    
    # Replace carrier IDs with average arrival delay
    
    X['op_unique_carrier'] = X['op_unique_carrier'].replace(carrierDict)
    
    # Find the average delay times by origin location, and store the values in
    # dictionary
    origin = pd.read_csv('origin_arr_delay.txt', delimiter = '\t', names =
                         ['origin', 'avg_delay'])
    origin = pd.Series(origin.avg_delay.values, index = origin.origin).to_dict()
    
    # Find the average delay times by destination location, and store the values in
    # a dictionary
    dest = pd.read_csv('dest_arr_delay.txt', delimiter = '\t', names = ['dest',
                        'avg_delay'])
    dest = pd.Series(dest.avg_delay.values, index = dest.dest).to_dict()
    
    # Replace the values in the "origin" and "dest" columns with the average arrival
    # delay time
    X['origin'] = X['origin'].replace(origin)
    X['dest'] = X['dest'].replace(dest)
    
    # Scale the numerical columns with one of the four scalers indicated above
    col_list = ['op_unique_carrier', 'origin', 'dest']
    X[col_list] = scaler.fit_transform(X[col_list])
    
    return X

def addXGBClsfPred(X,y_cat, pred_only=True):
    '''
    Adds to X, a new column predicting categorical value of y when given y_cat
    
    Take in X, a numerical only DataFrame, and y_cat, an arbitrary classification of y
    
    Option to get prediction only, or appended to X with pred_only
    '''
    
    from xgboost import XGBClassifier
    xgboost = XGBClassifier(n_estimators=100,learning_rate=0.1,reg_alpha=8)
    xgboost.fit(X, y_cat)
    y= xgboost.predict(X)
    if pred_only:
        return y
    else:
        X['xgbPred']=y
        return X

def replace_fl_date_with_num(X):
    '''
    Replaces fl_date column, a pandas datetime, with columns month and day of the week, integers
    Drops fl_date
    '''
    X['month'] = pd.DatetimeIndex(pd.to_datetime(X['fl_date'],
                        infer_datetime_format = True)).month
    X['dayWeek'] = pd.DatetimeIndex(pd.to_datetime(X['fl_date'],
                        infer_datetime_format = True)).dayofweek
    X.drop(columns='fl_date', inplace=True)
    return X