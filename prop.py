# coding: utf-8

### importing python libraries
import urllib, json, csv, sys, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import warnings
from datetime import datetime
from fbprophet import Prophet
warnings.filterwarnings("ignore")


### Getting started with data
data = sys.argv[1]
#data = json.dumps(data)
# decoding the JSON to dictionay
#d = json.loads(data)
# converting to dataframe
#dataset = pd.DataFrame.from_dict((d))
try:
    dataset = pd.read_excel(data)
except IOError as e:
    error = {"Status":"FALSE","Code":401,"Message":"Invalid file location.","Data":"NULL"}
    print (error)
    sys.exit()

cols = ['time_date','Values']
for col in cols:
        try:
            dataset[col]
        except KeyError as e:
            e = str(e)
            error = {"Status":"FALSE","Code":401,"Message":"Wrong column names: " + e, "Data":"NULL"}
            print (error)
            sys.exit()

### User input

#user = json.loads(sys.argv[2])
try:
    inp = int(sys.argv[2])
except ValueError as e:
    e = str(e)
    error = {"Status":"FALSE","Code":401,"Message":e + ". The number of next observations should be an integer.","Data":"NULL"}
    print (error)
    sys.exit()
            
dataset = dataset.dropna()

'''for i in dataset['time_date']:
    try:
        orig_date = str(i)
        datetime.strptime(orig_date, '%Y-%m-%d %H:%M:%S')
    except ValueError:
        error = {"Status":"FALSE","Code":401,"Message":"Incorrect data format, should be YYYY-MM-DD HH:MM:SS.","Data":"NULL"}
        print (error)
        sys.exit()
'''

dataset['time_date']= pd.to_datetime(dataset['time_date'])
train = dataset.iloc[0:-10,:]
test = dataset.iloc[-10:][['time_date','Values']]
train = train[['time_date','Values']]
train = train[train['Values']!=0]
train.index = train['time_date']
length = len(train)

### Gap between two times
interval = dataset['time_date'].iloc[1] - dataset['time_date'].iloc[0]

### Fitting the model in Prophet
X = pd.DataFrame(index=range(0,len(train)))
X['ds'] = train.index
X['y'] = train['Values'].values

m = Prophet()
m.fit(X)
future = m.make_future_dataframe(periods= inp + (len(dataset)-len(train)), freq = interval)
ff = m.predict(future)

### Calculating RMSE
rmse_test = ff[len(train):len(dataset)]
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(rmse_test['yhat'],test['Values'])
rmse = np.sqrt(mse)

predictions = pd.DataFrame(ff[len(dataset):]['yhat'].values, columns = {'Predicted'})
prediction = predictions.to_dict('records')
diction = {'RMSE': rmse, 'Predictions':prediction}
print(diction)