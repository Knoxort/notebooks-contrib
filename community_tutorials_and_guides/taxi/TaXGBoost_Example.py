#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#https://github.com/rapidsai-community/notebooks-contrib/blob/main/community_tutorials_and_guides/taxi/NYCTaxi-E2E.ipynb
#https://jovian.ai/allenkong221/nyc-taxi-fare/v/1?utm_source=embed#C2


# In[1]:


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib
from scipy import stats
from scipy.stats import norm, skew
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor, plot_importance
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
#import lightgbm as lgbm
import xgboost as xgb
import time


# In[2]:


'''if you get 'ModuleNotFoundError: No module named 'gcsfs', run `!pip install gcsfs` 
'''
#base_path = 'gcs://anaconda-public-data/nyc-taxi/csv/'

#df_2014 = dask_cudf.read_csv(base_path+'2014/yellow_*.csv')


#Ok, we need to load in data here, but not the old way
data_path = '/home/u117430/rapids/data/'
#df_2014 = pd.read_csv(data_path+'2014/yellow_*.csv')
df_2014 = pd.read_csv(data_path+'yellow_tripdata_2014-11.csv')
#df_2015 = pd.read_csv(data_path+'yellow_tripdata_2015-11.csv')
#df_2016 = pd.read_csv(data_path+'yellow_tripdata_2016-11.csv')
    #Sources:
        #https://github.com/oneapi-src/oneAPI-samples/blob/master/AI-and-Analytics/End-to-end-Workloads/Census/census_modin.ipynb
        #:https://examples.dask.org/dataframes/01-data-access.html#Read-CSV-files


# In[3]:


#Dictionary of required columns and their datatypes
must_haves = {
     ' pickup_datetime': 'datetime64[s]',
     ' dropoff_datetime': 'datetime64[s]',
     ' passenger_count': 'int32',
     ' trip_distance': 'float32',
     ' pickup_longitude': 'float32',
     ' pickup_latitude': 'float32',
     ' rate_code': 'int32',
     ' dropoff_longitude': 'float32',
     ' dropoff_latitude': 'float32',
     ' fare_amount': 'float32'
    }


# In[4]:


def clean(ddf, must_haves):
    tmp = {col:col.strip().lower() for col in list(ddf.columns)}     # replace the extraneous spaces in column names and lower the font type
        #In this case, what is tmp? It looks like tmp is jit dictionary built to hold the column names that have been fed in, but stripped of spaces and lower cased
    ddf = ddf.rename(columns=tmp) #Then, this dictionary is used to rename the columns
        
        #Rename documentionation: https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.rename.html

    ddf = ddf.rename(columns={
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'ratecodeid': 'rate_code'
    })  #More name changing. Just changing column names to an easier to read format
    
    ddf['pickup_datetime'] = ddf['pickup_datetime'].astype('datetime64[ms]')       #Looks to just recast datatype to a date/time format
    ddf['dropoff_datetime'] = ddf['dropoff_datetime'].astype('datetime64[ms]')

        #Astype doc: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.astype.html
    
    #Here's where things get tricky. Let's look at df.map_partitions() vs df.apply()
    
        #DataFrame.map_partitions(func, *args, **kwargs)
            #Desc: Apply Python function on each DataFrame partition.
            #Doc: https://docs.dask.org/en/latest/generated/dask.dataframe.DataFrame.map_partitions.html#dask.dataframe.DataFrame.map_partitions
        
        #DataFrame.apply(func, axis=0, raw=False, result_type=None, args=(), **kwargs)
            #Desc: Apply a function along an axis of the DataFrame.

        #So apply may not be what we want. map_partitions works on partitions, while apply works on axis
            #FYI: apply apparently shouldn't be used b/c it's horribly inefficient
            #DASK dataframes are made up of partitions, which are pandas dataframes?
                #https://docs.dask.org/en/latest/dataframe-best-practices.html
                #https://docs.dask.org/en/latest/dataframe-design.html#dataframe-design-partitions

    
    for col in ddf.columns:                                   #For each column
        if col not in must_haves:                             #If the column isn't in the dictionary
            ddf = ddf.drop(columns=col)                       #Remove it
            continue
        # if column was read as a string, recast as float
        if ddf[col].dtype == 'object':                        #If the column was a string
            ddf[col] = ddf[col].str.fillna('-1')              #Turn it into a float with these two lines
            ddf[col] = ddf[col].astype('float32')
        else:
            # downcast from 64bit to 32bit types
            # Tesla T4 are faster on 32bit ops
            if 'int' in str(ddf[col].dtype):                 #Convert int's to 32 bit ints
                ddf[col] = ddf[col].astype('int32')
            if 'float' in str(ddf[col].dtype):               #Convert doubles to floats
                ddf[col] = ddf[col].astype('float32')
            ddf[col] = ddf[col].fillna(-1)
    
    return ddf


# In[5]:


#df_2014 = df_2014.map_partitions(clean, must_haves, meta=must_haves)
taxi_df = df_2014
tmp = {col:col.strip().lower() for col in list(taxi_df.columns)}
taxi_df = taxi_df.rename(columns=tmp) #Then, this dictionary is used to rename the columns

taxi_df = taxi_df.rename(columns={
        'tpep_pickup_datetime': 'pickup_datetime',
        'tpep_dropoff_datetime': 'dropoff_datetime',
        'ratecodeid': 'rate_code'
    }) 

taxi_df['pickup_datetime'] = taxi_df['pickup_datetime'].astype('datetime64[ms]')       #Looks to just recast datatype to a date/time format
taxi_df['dropoff_datetime'] = taxi_df['dropoff_datetime'].astype('datetime64[ms]')


# In[6]:


taxi_df.head()


# In[7]:


#taxi_df = dask.dataframe.multi.concat([df_2014, df_2015, df_2016])
#taxi_df = pd.concat([df_2014, df_2016])
#taxi_df = df_2014


# In[8]:


#taxi_df = taxi_df.persist()


# In[9]:


final_taxi_df = taxi_df.drop(['vendor_id', 'store_and_fwd_flag', 'payment_type'], axis=1)


# In[10]:


#since we calculated the h_distance let's drop the trip_distance column, and then do model training with XGB.
#taxi_df = taxi_df.drop('trip_distance', axis=1)


# In[11]:


final_taxi_df.head()


# In[12]:


## add features

taxi_df['hour'] = taxi_df['pickup_datetime'].dt.hour
taxi_df['year'] = taxi_df['pickup_datetime'].dt.year
taxi_df['month'] = taxi_df['pickup_datetime'].dt.month
taxi_df['day'] = taxi_df['pickup_datetime'].dt.day
taxi_df['day_of_week'] = taxi_df['pickup_datetime'].dt.weekday
taxi_df['is_weekend'] = (taxi_df['day_of_week']>=5).astype('int32')

#calculate the time difference between dropoff and pickup.
taxi_df['diff'] = taxi_df['dropoff_datetime'].astype('int64') - taxi_df['pickup_datetime'].astype('int64')
taxi_df['diff']=(taxi_df['diff']/1000).astype('int64')

taxi_df['pickup_latitude_r'] = taxi_df['pickup_latitude']//.01*.01
taxi_df['pickup_longitude_r'] = taxi_df['pickup_longitude']//.01*.01
taxi_df['dropoff_latitude_r'] = taxi_df['dropoff_latitude']//.01*.01
taxi_df['dropoff_longitude_r'] = taxi_df['dropoff_longitude']//.01*.01

#taxi_df = taxi_df.drop('pickup_datetime', axis=1)
#taxi_df = taxi_df.drop('dropoff_datetime', axis=1)


# In[13]:


for col in taxi_df.columns:
    print(col)


# In[14]:


final_taxi_df = taxi_df.drop(['pickup_datetime','dropoff_datetime','vendor_id', 'store_and_fwd_flag', 'payment_type'], axis=1)


# In[15]:


X, y = final_taxi_df.drop('fare_amount', axis = 1), final_taxi_df['fare_amount']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)


# In[16]:


dtrain = xgb.DMatrix(X_train, label=y_train)


# In[17]:


dvalid = xgb.DMatrix(X_test, label=y_test)


# In[18]:


dtest = xgb.DMatrix(final_taxi_df)


# In[19]:


watchlist = [(dtrain, 'train'), (dvalid, 'valid')]


# In[20]:


xgb_params = {
    #'tree_method' : 'hist',
    'updater' : 'grow_quantile_histmaker_oneapi',
    'min_child_weight': 1, 
    'learning_rate': 0.05, 
    'colsample_bytree': 0.7, 
    'max_depth': 10,
    'subsample': 0.7,
    'n_estimators': 5000,
    'n_jobs': -1, 
    'booster' : 'gbtree', 
    'silent': 1,
    'eval_metric': 'rmse'}


# In[ ]:

tmp = time.time()
model = xgb.train(xgb_params, dtrain, 700, watchlist, early_stopping_rounds=100, maximize=False, verbose_eval=50)
print("xgb.train time: %s seconds" % (str(time.time() - tmp)))


# In[ ]:

tmp = time.time()
y_train_pred = model.predict(dtrain)
print("model.predict(dtrain) time: %s seconds" % (str(time.time() - tmp)))

tmp = time.time()
y_pred = model.predict(dvalid)
print("model.predict(dvalid) time: %s seconds" % (str(time.time() - tmp)))


print('Train r2 score: ', r2_score(y_train_pred, y_train))
print('Test r2 score: ', r2_score(y_test, y_pred))
train_rmse = np.sqrt(mean_squared_error(y_train_pred, y_train))
test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Train RMSE: {train_rmse:.4f}')
print(f'Test RMSE: {test_rmse:.4f}')


# In[ ]:




