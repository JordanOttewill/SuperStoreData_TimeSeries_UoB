# -*- coding: utf-8 -*-
"""
Created on Sun Jul 24 21:23:31 2022

@author: jordan.ottewill
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

#import csv
SuperStoreMaster = pd.read_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\train.csv')
SuperStore = pd.read_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\train.csv')

#info
print(SuperStore.info())
print(SuperStore.describe())

print(SuperStore['Product ID'].nunique())
print(SuperStore['Product Name'].nunique())
name_error = (SuperStore[SuperStore["Product Name"]=="Staple envelope"])
print(name_error['Product ID'].nunique())
print(SuperStore.groupby("Category").Sales.sum())

plt.figure()
sns.histplot(data=name_error, x='Sales')
plt.title('Histogram of Staple Envelope sales')
plt.show()

regionx = SuperStore['Region'].unique()
regiony = SuperStore.groupby("Region").Sales.sum()
##11 postal code value missing

plt.figure()
plt.bar(regionx[2],regiony[0],label='Central',color='blue')
plt.bar(regionx[3],regiony[1],label='East',color='orange')
plt.bar(regionx[0],regiony[2],label='South',color='red')
plt.bar(regionx[1],regiony[3],label='West',color='green')
plt.title('Sum of sales by Region')
plt.ylabel('Sum of Sales')
plt.xlabel('Region')
for i, v in enumerate(regiony):
    plt.text(i-0.2,
             i+100,
             round(regiony[i]),
             fontsize=10,
             color='white')
plt.show()

categoryx = SuperStore['Category'].unique()
categoryy = SuperStore.groupby("Category").Sales.sum()
categorycount = SuperStore['Category'].value_counts()
##11 postal code value missing

plt.figure()
plt.bar(categoryx[1],categoryy[1],label='Office Supplies',color='green')
plt.bar(categoryx[0],categoryy[0],label='Furniture',color='blue')
plt.bar(categoryx[2],categoryy[2],label='Technology',color='red')
plt.title('Sales by Category with Count of Sales overlay')
plt.ylabel('Sum of Sales')
plt.xlabel('Category')
for i, v in enumerate(categorycount):
    plt.text(i-0.2,
             i+100,
             round(categorycount[i]),
             fontsize=10,
             color='white')
plt.show()
###############################################################################
##Data cleanse

NaNPostal = []

for x,y in SuperStore.iterrows():
    rowhasNaN = y.isnull()
    if rowhasNaN.any():
        NaNPostal.append(x)

NaNValues = SuperStore.iloc[NaNPostal,:]
#view rows with missing values.
##compare like rows

print(SuperStore[(SuperStore['City']=='Burlington')&(SuperStore['State']=='Vermont')])
print(SuperStore[(SuperStore['Region']=='East')&(SuperStore['State']=='Vermont')])
##only the missing values

"""The missing values, postal code, will not impact forecasting as it is the postal code of the store.
 City, and State can be used for this same purpose instead. Column Postal Code to be dropped."""
 
SuperStore.drop(columns='Postal Code',inplace=True)

##Other candidate for removal are: 
    #product name
    #customer name
    #country (Not relevant)
    #analysis needed on region/city/state
    #rowid
    #ship date
    
#Location analysis
print(SuperStore.loc[:,['City','State','Region']].drop_duplicates())
#There are 600 rows, meaning 600 distinct combinations. Therefore each level of reference is needed.

"""Convert Date fields to datetime objects"""
SuperStore['Order Date'] = pd.to_datetime(SuperStore['Order Date'])
SuperStore['Ship Date'] = pd.to_datetime(SuperStore['Ship Date'])

"""Create feature Dwell""" ##Analyse order_ship_date quality.
dwell = []

print(SuperStore.loc[:,['Order Date','Ship Date']])

for x in np.arange(0,len(SuperStore.loc[:,['Order Date','Ship Date']])):
    dwell.append(SuperStore.loc[x,'Ship Date']-SuperStore.loc[x,'Order Date'])

SuperStore['Dwell'] = dwell
##there are negative dwell values. Drop shp date and dwell. Only focus on Order Date.

#SuperStore.to_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStorev1.csv')

"""Visualise Trends"""

plt.plot()
sns.scatterplot(data=SuperStore, x='Order Date', y = 'Sales', hue = 'Category')
plt.show()

"""Some outliers in tech?"""
print(SuperStore[SuperStore['Category']=='Technology'])

plt.plot()
sns.scatterplot(data=SuperStore[SuperStore['Category']=='Technology'], x='Order Date', y = 'Sales', hue = 'Sales')
plt.title('Technology sales')
plt.show()
#Not necessarily and outlier, BUT the only one they sold in the product ID. 
#Clearly very expensive and the only one they have ever sold. Unlikely to sell again so removing might make sense.
#Not representative of true forecast.
#inex 2697
print(SuperStore.iloc[2697,:])
SuperStore.drop([2697],inplace=True)


##############################################################################
###feature creation
##variable transformation...log of sales?

"""ID's are favoured over names, reduces mining complexity"""
SuperStore.drop(columns=['Customer Name','Product Name'],inplace=True)
"""No information gained by Country...only 1 value"""
SuperStore.drop(columns='Country',inplace=True)
"""RowID not valid"""
SuperStore.drop(columns='Row ID',inplace=True)
"""Order ID is always a distinct value...no information gained"""
SuperStore.drop(columns='Order ID',inplace=True)

##Round 1 we drop all grouping
SuperStore.drop(columns=['Customer ID','Segment','City','State','Region','Product ID','Category','Sub-Category','Dwell','Ship Mode','Ship Date'],inplace=True)

"""In order to forecast we need to decide the level of aggregation..."""
SuperStore = SuperStore.groupby(by='Order Date', as_index=True).sum()

"""Reorder by index"""
SuperStore.sort_index(inplace=True)

#"""Convert ship Mode to numeric values??"""
"""for x in np.arange(0,len(SuperStore['Ship Mode'])):
    if SuperStore.loc[x,'Ship Mode'] == 'First Class':
        SuperStore.loc[x,'Ship Mode'] = 1
    elif SuperStore.loc[x,'Ship Mode'] == 'Second Class':
        SuperStore.loc[x,'Ship Mode'] = 2
    elif SuperStore.loc[x,'Ship Mode'] == 'Standard Class':
        SuperStore.loc[x,'Ship Mode'] = 3
    elif SuperStore.loc[x,'Ship Mode'] == 'Same Day':
        SuperStore.loc[x,'Ship Mode'] = 0
SuperStore['Ship Mode'] = SuperStore['Ship Mode'].astype(int)"""

"""DateTime Features"""
##print(SuperStore['Order Date'])
##SuperStore.set_index('Order Date',drop = True, inplace = True)
SuperStore['Year'] = SuperStore.index.year
SuperStore['Month'] = SuperStore.index.month
SuperStore['Day of Year'] = SuperStore.index.dayofyear
SuperStore['Day of Month'] = SuperStore.index.day
SuperStore['Day of Week'] = SuperStore.index.dayofweek
SuperStore['Week of Year'] = SuperStore.index.isocalendar().week.astype(int)


"""Create Lag features"""
##Lag features only work on the grouped data...
target_map = SuperStore['Sales'].to_dict()
SuperStore['Lag1'] = (SuperStore.index - pd.Timedelta('364 days')).map(target_map)
SuperStore['Lag2'] = (SuperStore.index - pd.Timedelta('728 days')).map(target_map)
SuperStore['Lag3'] = (SuperStore.index - pd.Timedelta('1092 days')).map(target_map)
###364 is divisible by 7 so will get same day of week.

"""Data ready for forecasting"""
"""Train Test Split"""
print('HEAD: ',SuperStore.head())
print('TAIL: ',SuperStore.tail())
##2nd Jan 15 till 30 Dec 18
##xgboost imported
##default settings

reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.3,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       ##n_estimators = ,
                       ##early_stopping_rounds = ,
                       seed = 1)

"""Train = 3.5 years, test = last 6 months"""
tr_data = SuperStore[SuperStore.index < '2018-06-01 00:00:00']
te_data = SuperStore[SuperStore.index >= '2018-06-01 00:00:00']
#train
tr_features = tr_data.iloc[:,1:10]
tr_target = tr_data.iloc[:,0]
#test
te_features = te_data.iloc[:,1:10]
te_target = te_data.iloc[:,0]

"""Show train test split"""
sns.lineplot(x=tr_target.index, y=tr_target, label='Train')
sns.lineplot(x=te_target.index, y=te_target, label = 'Test')
plt.legend()
plt.title("Initial Train Test Split")
plt.show()

"""train model"""
reg.fit(tr_features,tr_target,
        eval_set = [(tr_features,tr_target),(te_features,te_target)],
        verbose = True #get readouts as fitting
        )
"""Predict sales"""
sales_pred = reg.predict(te_features)

"""Compare test to forecast"""
plt.figure()
fig = sns.lineplot(x=te_target.index, y=te_target, label = 'Test')
sns.lineplot(x=te_target.index, y=sales_pred, label = 'Forecast')
fig.spines['bottom'].set_position('zero')
fig.tick_params(axis='x', pad = 20)
plt.title('learning rate 0.3, 99 iterations, no stopping')
plt.legend()
plt.show()

evals_result_1 = reg.evals_result()

plt.figure(figsize=(10,7))
plt.plot(evals_result_1["validation_0"]["rmse"], label="Training loss")
plt.plot(evals_result_1["validation_1"]["rmse"], label="Validation loss")
plt.axvline(10, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.title('Initial Validation and Training RMSE values during model fitting')
plt.ylabel("Loss")
plt.legend()

"""RMSE"""
"""output from the xgboosts algorithm shows around 10th iteration, the RMSE of the test dataset starts to increase, while the RMSE of the training set continures to decrease.
This suggests the model starts to overfit to the training data."""
"""We can decrease the learning rate to arrive at the minimum more carefully and also not overshoot it."""

#Decrease lrate
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 1000,   ##use a large number of trees to find the best number
                       ##early_stopping_rounds = ,
                       seed = 1)

"""train model"""
reg.fit(tr_features,tr_target,
        eval_set = [(tr_features,tr_target),(te_features,te_target)],
        verbose = 10 #get readouts as fitting
        )
"""Predict sales"""
sales_pred = reg.predict(te_features)

"""Compare test to forecast"""
plt.figure()
fig = sns.lineplot(x=te_target.index, y=te_target, label = 'Test')
sns.lineplot(x=te_target.index, y=sales_pred, label = 'Forecast')
fig.spines['bottom'].set_position('zero')
fig.tick_params(axis='x', pad = 20)
plt.title('learning rate 0.01, 1000 iterations, no stopping')
plt.legend()
plt.show()

evals_result_2 = reg.evals_result()

plt.figure(figsize=(10,7))
plt.plot(evals_result_2["validation_0"]["rmse"], label="Training loss")
plt.plot(evals_result_2["validation_1"]["rmse"], label="Validation loss")
plt.axvline(206, color="gray", label="Optimal tree number")
plt.xlabel("Number of trees")
plt.ylabel("Loss")
plt.title('Initial Validation and Training RMSE values during model fitting - second iteration')
plt.legend()
"""Smoother curve as learning rate is smaller"""

"""Introduce early stopping to get the BEST tree. Currently we are overfitting."""
#Increase lrate
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 1000,   ##use a large number of trees to find the best number
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1)

"""train model"""
reg.fit(tr_features,tr_target,
        eval_set = [(tr_features,tr_target),(te_features,te_target)],
        verbose = 10 #get readouts as fitting
        )
"""Predict sales"""
sales_pred = reg.predict(te_features)

print(reg.best_ntree_limit)
##295

"""Compare test to forecast"""
plt.figure()
fig = sns.lineplot(x=te_target.index, y=te_target, label = 'Test')
sns.lineplot(x=te_target.index, y=sales_pred, label = 'Forecast')
fig.spines['bottom'].set_position('zero')
fig.tick_params(axis='x', pad = 20)
plt.title('learning rate 0.01, early stopping at 295 trees')
plt.legend()
plt.show()

evals_result_3 = reg.evals_result()

##Manual MSE/RMSE:
MSE = np.mean((sales_pred - te_target)  ** 2)
RMSE = np.sqrt(MSE)
print("RMSE: ", RMSE)
##2530.1009
print(evals_result_3["validation_1"]["rmse"][-1])
##2531.3802

print(SuperStore.Sales.mean())

#We have initial Model. Find optimal split of data.
##ycle through train test sizes to fid the correct size to use?
indices = SuperStore.index
#test
print(indices[-(7*4)])
print(728/7)
##104 weeks in 2 years.
#test
print(SuperStore[SuperStore.index < indices[-(7*1)]])
print(evals_result_3["validation_1"]["rmse"][-1])

rmse = []
splt = []
ntrees = []
split = 1
#create empty dictionary
predictions = {}

    
while split < 105:
    #TR TE Split
    tr_data = SuperStore[SuperStore.index < indices[-(7*split)]]
    te_data = SuperStore[SuperStore.index >= indices[-(7*split)]]
    #train
    tr_features = tr_data.iloc[:,1:10]
    tr_target = tr_data.iloc[:,0]
    #test
    te_features = te_data.iloc[:,1:10]
    te_target = te_data.iloc[:,0]
    #Fit model (Same model applied each time.)
    reg.fit(tr_features,tr_target,
            eval_set = [(tr_features,tr_target),(te_features,te_target)],
            verbose = False
            )
    #Prediction using predict fxn
    sales_pred = reg.predict(te_features)
    # RMSE (error)
    evals_result = reg.evals_result()
    rmse.append(evals_result["validation_1"]["rmse"][-1])
    splt.append(split) #number of weeks tested on
    ntrees.append(reg.best_ntree_limit) #number of trees used
    predictions[split] = sales_pred #store each prediction set.
    split = split+1
    print("split ",split," done")
    
split_decision = {
    "Split": splt,
    "RMSE": rmse,
    "nTrees": ntrees
    }

splitdecision = pd.DataFrame(split_decision)

plt.figure()
sns.scatterplot(data = splitdecision, x = 'Split', y = 'RMSE', hue = 'nTrees')
plt.title('RMSE of XGBoost for x week predictions and number of trees made')
plt.xlabel('Number of weeks forecast')
plt.xticks(np.arange(0,110, step =10))
plt.axvline(27.5, color="gray", label="Initial Model forecast")
plt.legend(loc=4)
plt.show()

##33 and 36 are good. 33 has less trees.
#Visualise 33
##
#TR TE Split - using 36 splits per above findngs
tr_data = SuperStore[SuperStore.index < indices[-(7*33)]]
te_data = SuperStore[SuperStore.index >= indices[-(7*33)]]
#train
tr_features = tr_data.iloc[:,1:10]
tr_target = tr_data.iloc[:,0]
#test
te_features = te_data.iloc[:,1:10]
te_target = te_data.iloc[:,0]

#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 1000,   ##use a large number of trees to find the best number
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1)

"""train model"""
reg.fit(tr_features,tr_target,
        eval_set = [(tr_features,tr_target),(te_features,te_target)],
        verbose = 10 #get readouts as fitting
        )
"""Predict sales"""
sales_pred = reg.predict(te_features)

print(reg.best_ntree_limit)
##193

opt_evals_result = reg.evals_result()
print(opt_evals_result["validation_1"]["rmse"][-1])
##2309.59

"""Compare test to forecast"""
plt.figure()
fig = sns.lineplot(x=te_target.index, y=te_target, label = 'Test')
sns.lineplot(x=te_target.index, y=sales_pred, label = 'Forecast')
fig.spines['bottom'].set_position('zero')
fig.tick_params(axis='x', pad = 20)
plt.title('learning rate 0.01, 33 week prediction, 193 trees. Optimum review length')
plt.legend()
plt.show()

"""Comparison of whole set"""
plt.figure()
fig = sns.lineplot(x=SuperStore.index, y=SuperStore.Sales, label = 'Test')
sns.lineplot(x=te_target.index, y=sales_pred, label = 'Forecast')
fig.spines['bottom'].set_position('zero')
fig.tick_params(axis='x', pad = 20)
plt.title('Forecast against entire dataset')
plt.legend()
plt.show()
##33 weeks outward gives best RMSE...

"""Most important features"""
print(reg.feature_names_in_, "importance:", reg.feature_importances_)

"""Forecast future"""
##231 days (7*33 was optimum forecast out)
#create empty future cells.
print(indices[-1])  #max date

date_range = pd.date_range('2018-12-30 00:00:00', '2019-12-30 00:00:00',freq='D')

future_data = pd.DataFrame({'Order Date':date_range})
future_data['Sales'] = ''
future_data.set_index('Order Date',drop = True, inplace = True)
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day of Year'] = future_data.index.dayofyear
future_data['Day of Month'] = future_data.index.day
future_data['Day of Week'] = future_data.index.dayofweek
future_data['Week of Year'] = future_data.index.isocalendar().week.astype(int)

"""Create Lag features"""
target_map = SuperStore['Sales'].to_dict()
future_data['Lag1'] = (future_data.index - pd.Timedelta('364 days')).map(target_map)
future_data['Lag2'] = (future_data.index - pd.Timedelta('728 days')).map(target_map)
future_data['Lag3'] = (future_data.index - pd.Timedelta('1092 days')).map(target_map)
###364 is divisible by 7 so will get same day of week.

""""Data Setup"""
#train
known_features = SuperStore.iloc[:,1:10]
known_sales = SuperStore.iloc[:,0]
#test
future_features = future_data.iloc[:,1:10]

"""best number of trees"""
print(splitdecision.nTrees.mean())

"""Model tune to avoid overfit...No test set to validate on"""
#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 243,   ##Av. no of trees was 243, we have no validation set for futrue data so have to control end.
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1)

"""train model"""
reg.fit(known_features,known_sales,
        eval_set = [(known_features,known_sales)],
        verbose = 10 #get readouts as fitting
        )

"""Predict sales"""
futures_sales = reg.predict(future_features)

"""Comparison of whole set"""
plt.figure()
fig = sns.lineplot(x=SuperStore.index, y=SuperStore.Sales, label = 'Test')
sns.lineplot(x=future_features.index, y=futures_sales, label = 'Forecast')
fig.spines['bottom'].set_position('zero')
fig.tick_params(axis='x', pad = 20)
plt.title('Sales with forecast for the next year')
plt.legend()
plt.show()

evals_result_future = reg.evals_result()
print(evals_result_future["validation_0"]["rmse"][-1])
##1555.92

"""Can we improve rmse with cross validation?"""
from sklearn.model_selection import TimeSeriesSplit

timeseriessplit = TimeSeriesSplit(n_splits = 3,  #data runs from 2015 so cannot go back further
                                  test_size = 300, ##1230 rows meaning we can't have larger splits
                                  gap = 1)

timeseriessplit.split(SuperStore)

"""Visulaise splits"""

for train_index, validation_index in timeseriessplit.split(SuperStore):
    print("Train Index: ", train_index, "n/, Validation Index: ", validation_index )
    #print(SuperStore.iloc[train_index,0])
    #print(indices[validation_index.min()])
    
fig, subplot = plt.subplots(3,1, sharex = True)
    
fold = 0
for train_index, validation_index in timeseriessplit.split(SuperStore):
    sns.lineplot(ax = subplot[fold],
                 data = SuperStore.iloc[train_index,0],
                 #x = index,
                 #y = 'Sales',
                 label = 'Training',
                 )
    sns.lineplot(ax = subplot[fold],
                 data = SuperStore.iloc[validation_index,0],
                 label = 'Test')
    subplot[fold].axvline(indices[validation_index.min()], color = 'gray')
    subplot[fold].set_title("Train_Test Split Fold "+str(fold+1))
    fold = fold+1
plt.show()

"""Test splits"""

#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 500,   ##Now we have validation set, can utilise early stopping.
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1
                       )

scores = []
folds = []
trees = []
fold = 1

for train_index, validation_index in timeseriessplit.split(SuperStore):
    tr_data = SuperStore.iloc[train_index]
    te_data = SuperStore.iloc[validation_index]
    #train
    tr_features = tr_data.iloc[:,1:10]
    tr_target = tr_data.iloc[:,0]
    #test
    te_features = te_data.iloc[:,1:10]
    te_target = te_data.iloc[:,0]
    """train model"""
    reg.fit(tr_features,tr_target,
            eval_set = [(tr_features,tr_target),(te_features,te_target)],
            verbose = 10 #get readouts as fitting
            )
    sales_pred = reg.predict(te_features)
    evals_result = reg.evals_result()
    scores.append(evals_result["validation_1"]["rmse"][-1])
    trees.append(reg.best_ntree_limit)
    folds.append(fold)
    fold = fold+1
    
print(scores)
print(np.mean(scores))
##2332.7714
##NOT an improvement on the K-Fold validation number...

import pickle
pickle.dump(reg, open(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreModel.sav','wb'))

###############################################################################
##Can we forecast the sales for each category???

SuperStoreCat = SuperStoreMaster.copy(deep = True)

SuperStoreCat.drop(columns='Postal Code',inplace=True)

"""Outliers in tech?"""
#Not necessarily and outlier, BUT the only one they sold in the product ID. 
#Clearly very expensive and the only one they have ever sold. Unlikely to sell again so removing might make sense.
#Not representative of true forecast.
#inex 2697
print(SuperStoreCat.iloc[2697,:])
SuperStoreCat.drop([2697],inplace=True)

"""Convert Date fields to datetime objects"""
SuperStoreCat['Order Date'] = pd.to_datetime(SuperStoreCat['Order Date'])

"""ID's are favoured over names, reduces mining complexity"""
SuperStoreCat.drop(columns=['Customer Name','Product Name'],inplace=True)
"""No information gained by Country...only 1 value"""
SuperStoreCat.drop(columns='Country',inplace=True)
"""RowID not valid"""
SuperStoreCat.drop(columns='Row ID',inplace=True)
"""Order ID is always a distinct value...no information gained"""
SuperStoreCat.drop(columns='Order ID',inplace=True)
##Round 2 we drop all except category
SuperStoreCat.drop(columns=['Customer ID','Segment','City','State','Region','Product ID','Sub-Category','Ship Mode','Ship Date'],inplace=True)

"""In order to forecast we need to decide the level of aggregation..."""
SuperStoreCat = SuperStoreCat.groupby(by=['Order Date','Category'],as_index=False).sum()

"""Reorder by index"""
SuperStoreCat.set_index('Order Date',drop = True,inplace = True)
SuperStoreCat.sort_index(inplace=True)

"""DateTime Features"""
##print(SuperStore['Order Date'])
SuperStoreCat['Year'] = SuperStoreCat.index.year
SuperStoreCat['Month'] = SuperStoreCat.index.month
SuperStoreCat['Day of Year'] = SuperStoreCat.index.dayofyear
SuperStoreCat['Day of Month'] = SuperStoreCat.index.day
SuperStoreCat['Day of Week'] = SuperStoreCat.index.dayofweek
SuperStoreCat['Week of Year'] = SuperStoreCat.index.isocalendar().week.astype(int)


"""Create Lag features"""
#Separate the three and then concat df.
tech_target_map = SuperStoreCat[SuperStoreCat['Category']=='Technology'].Sales.to_dict()
furniture_target_map = SuperStoreCat[SuperStoreCat['Category']=='Furniture'].Sales.to_dict()
office_target_map = SuperStoreCat[SuperStoreCat['Category']=='Office Supplies'].Sales.to_dict()

techdf = SuperStoreCat[SuperStoreCat['Category']=='Technology'].copy(deep=True)
furndf = SuperStoreCat[SuperStoreCat['Category']=='Furniture'].copy(deep=True)
offdf = SuperStoreCat[SuperStoreCat['Category']=='Office Supplies'].copy(deep=True)

techdf['Lag1'] = (techdf.index - pd.Timedelta('364 days')).map(tech_target_map)
techdf['Lag2'] = (techdf.index - pd.Timedelta('728 days')).map(tech_target_map)
techdf['Lag3'] = (techdf.index - pd.Timedelta('1092 days')).map(tech_target_map)
###364 is divisible by 7 so will get same day of week.
furndf['Lag1'] = (furndf.index - pd.Timedelta('364 days')).map(furniture_target_map)
furndf['Lag2'] = (furndf.index - pd.Timedelta('728 days')).map(furniture_target_map)
furndf['Lag3'] = (furndf.index - pd.Timedelta('1092 days')).map(furniture_target_map)

offdf['Lag1'] = (offdf.index - pd.Timedelta('364 days')).map(office_target_map)
offdf['Lag2'] = (offdf.index - pd.Timedelta('728 days')).map(office_target_map)
offdf['Lag3'] = (offdf.index - pd.Timedelta('1092 days')).map(office_target_map)

#Concat and re-sort
SuperStoreCat = pd.concat([techdf, offdf, furndf])
SuperStoreCat.sort_index(inplace=True)

"""Data Prep complete."""
"""Rearrange columns to have sales at front"""
print(SuperStoreCat.columns)

SuperStoreCat = SuperStoreCat[['Sales', 'Category', 'Year', 'Month', 'Day of Year', 'Day of Month',
       'Day of Week', 'Week of Year', 'Lag1', 'Lag2', 'Lag3']]

"""Cross Validation"""
#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 500,   ##Now we have validation set, can utilise early stopping.
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1,
                       enable_categorical=True,  ##allows categorical data
                       tree_method = 'hist' ##attempt to allow cat. data
                       )

##data type
SuperStoreCat['Category'] = SuperStoreCat['Category'].astype("category")
print(SuperStoreCat.info())

scores = []
folds = []
trees = []
fold = 1


for train_index, validation_index in timeseriessplit.split(SuperStoreCat):
    tr_data = SuperStoreCat.iloc[train_index]
    te_data = SuperStoreCat.iloc[validation_index]
    #train
    tr_features = tr_data.iloc[:,1:11]
    tr_target = tr_data.iloc[:,0]
    #test
    te_features = te_data.iloc[:,1:11]
    te_target = te_data.iloc[:,0]
    """train model"""
    reg.fit(tr_features,tr_target,
            eval_set = [(tr_features,tr_target),(te_features,te_target)],
            verbose = 10 #get readouts as fitting
            )
    sales_pred = reg.predict(te_features)
    evals_result = reg.evals_result()
    scores.append(evals_result["validation_1"]["rmse"][-1])
    trees.append(reg.best_ntree_limit)
    folds.append(fold)
    fold = fold+1
    
print(scores)
print(np.mean(scores))
##1363
#trees around 150

##applying tree method hist = rmse of 2332 on k-fold. SO improvement due to the introduction of categorical variables.

rmse = []
splt = []
ntrees = []
split = 1
#create empty dictionary
predictions = {}

    
while split < 105:
    #TR TE Split
    tr_data = SuperStoreCat[SuperStoreCat.index < indices[-(7*split)]]
    te_data = SuperStoreCat[SuperStoreCat.index >= indices[-(7*split)]]
    #train
    tr_features = tr_data.iloc[:,1:10]
    tr_target = tr_data.iloc[:,0]
    #test
    te_features = te_data.iloc[:,1:10]
    te_target = te_data.iloc[:,0]
    #Fit model (Same model applied each time.)
    reg.fit(tr_features,tr_target,
            eval_set = [(tr_features,tr_target),(te_features,te_target)],
            verbose = False
            )
    #Prediction using predict fxn
    sales_pred = reg.predict(te_features)
    # RMSE (error)
    evals_result = reg.evals_result()
    rmse.append(evals_result["validation_1"]["rmse"][-1])
    splt.append(split) #number of weeks tested on
    ntrees.append(reg.best_ntree_limit) #number of trees used
    predictions[split] = sales_pred #store each prediction set.
    split = split+1
    print("split ",split," done")
    
split_decisioncat = {
    "Split": splt,
    "RMSE": rmse,
    "nTrees": ntrees
    }

splitdecisioncat = pd.DataFrame(split_decisioncat)

plt.figure()
sns.scatterplot(data = splitdecisioncat, x = 'Split', y = 'RMSE', hue = 'nTrees')
plt.title('RMSE of XGBoost for x week predictions and number of trees made')
plt.xlabel('Number of weeks forecast')
plt.xticks(np.arange(0,110, step =10))
plt.axvline(27.5, color="gray", label="Initial Model forecast")
plt.legend(loc=4)
plt.show()

print(np.mean(scores))
print(SuperStoreCat.Sales.mean())

##33 optimal again.
#Visualise 33
##
#TR TE Split - using 36 splits per above findngs
tr_data = SuperStoreCat[SuperStoreCat.index < indices[-(7*33)]]
te_data = SuperStoreCat[SuperStoreCat.index >= indices[-(7*33)]]
#train
tr_features = tr_data.iloc[:,1:10]
tr_target = tr_data.iloc[:,0]
#test
te_features = te_data.iloc[:,1:10]
te_target = te_data.iloc[:,0]

reg.fit(tr_features,tr_target,
        eval_set = [(tr_features,tr_target),(te_features,te_target)],
        verbose = False
        )

sales_pred = reg.predict(te_features)

"""Compare test to forecast"""

#predictions = pd.Series(np.array(sales_pred).tolist()).copy(deep=True)
predictions = np.array(sales_pred).tolist()

te_data['Prediction'] = predictions

print(te_data[te_data['Category']=='Technology'])

plots, subplots = plt.subplots(3,1, sharex=True)
sns.lineplot(ax = subplots[0],
                   data = te_data[te_data['Category']=='Technology'],
                   x=te_data[te_data['Category']=='Technology'].index,
                   y='Sales', 
                   ##hue ='Category', 
                   label = 'Test')

sns.lineplot(ax = subplots[0],
             data = te_data[te_data['Category']=='Technology'],
             x=te_data[te_data['Category']=='Technology'].index,
             y='Prediction', 
             ##hue ='Category', 
             label = 'Forecast')

sns.lineplot(ax = subplots[1],
                   data = te_data[te_data['Category']=='Furniture'],
                   x=te_data[te_data['Category']=='Furniture'].index,
                   y='Sales', 
                   ##hue ='Category', 
                   label = 'Test')

sns.lineplot(ax = subplots[1],
             data = te_data[te_data['Category']=='Furniture'],
             x=te_data[te_data['Category']=='Furniture'].index,
             y='Prediction', 
             ##hue ='Category', 
             label = 'Forecast')

sns.lineplot(ax = subplots[2],
                   data = te_data[te_data['Category']=='Office Supplies'],
                   x=te_data[te_data['Category']=='Office Supplies'].index,
                   y='Sales', 
                   ##hue ='Category', 
                   label = 'Test')

sns.lineplot(ax = subplots[2],
             data = te_data[te_data['Category']=='Office Supplies'],
             x=te_data[te_data['Category']=='Office Supplies'].index,
             y='Prediction', 
             ##hue ='Category', 
             label = 'Forecast')
plt.legend()
plt.show()


"""Most important features"""
print(reg.feature_names_in_, "importance:", reg.feature_importances_)

pickle.dump(reg, open(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreCatModel.sav','wb'))

"""Forecast future"""
#create empty future cells.
print(indices[-1])  #max date

date_range = pd.date_range('2018-12-30 00:00:00', '2019-12-30 00:00:00',freq='D')

future_data = pd.DataFrame({'Order Date':date_range})
future_data['Sales'] = ''
future_data['Category'] = 'Technology'
future_data.set_index('Order Date',drop = True, inplace = True)
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day of Year'] = future_data.index.dayofyear
future_data['Day of Month'] = future_data.index.day
future_data['Day of Week'] = future_data.index.dayofweek
future_data['Week of Year'] = future_data.index.isocalendar().week.astype(int)

future_tech = future_data.copy(deep=True)

future_data = pd.DataFrame({'Order Date':date_range})
future_data['Sales'] = ''
future_data['Category'] = 'Furniture'
future_data.set_index('Order Date',drop = True, inplace = True)
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day of Year'] = future_data.index.dayofyear
future_data['Day of Month'] = future_data.index.day
future_data['Day of Week'] = future_data.index.dayofweek
future_data['Week of Year'] = future_data.index.isocalendar().week.astype(int)

future_furniture = future_data.copy(deep=True)

future_data = pd.DataFrame({'Order Date':date_range})
future_data['Sales'] = ''
future_data['Category'] = 'Office Supplies'
future_data.set_index('Order Date',drop = True, inplace = True)
future_data['Year'] = future_data.index.year
future_data['Month'] = future_data.index.month
future_data['Day of Year'] = future_data.index.dayofyear
future_data['Day of Month'] = future_data.index.day
future_data['Day of Week'] = future_data.index.dayofweek
future_data['Week of Year'] = future_data.index.isocalendar().week.astype(int)

future_office = future_data.copy(deep=True)

"""Create Lag features"""
#Separate the three and then concat df.
tech_target_map = SuperStoreCat[SuperStoreCat['Category']=='Technology'].Sales.to_dict()
furniture_target_map = SuperStoreCat[SuperStoreCat['Category']=='Furniture'].Sales.to_dict()
office_target_map = SuperStoreCat[SuperStoreCat['Category']=='Office Supplies'].Sales.to_dict()

future_tech['Lag1'] = (future_tech.index - pd.Timedelta('364 days')).map(tech_target_map)
future_tech['Lag2'] = (future_tech.index - pd.Timedelta('728 days')).map(tech_target_map)
future_tech['Lag3'] = (future_tech.index - pd.Timedelta('1092 days')).map(tech_target_map)

future_furniture['Lag1'] = (future_furniture.index - pd.Timedelta('364 days')).map(furniture_target_map)
future_furniture['Lag2'] = (future_furniture.index - pd.Timedelta('728 days')).map(furniture_target_map)
future_furniture['Lag3'] = (future_furniture.index - pd.Timedelta('1092 days')).map(furniture_target_map)

future_office['Lag1'] = (future_office.index - pd.Timedelta('364 days')).map(office_target_map)
future_office['Lag2'] = (future_office.index - pd.Timedelta('728 days')).map(office_target_map)
future_office['Lag3'] = (future_office.index - pd.Timedelta('1092 days')).map(office_target_map)

future_df = pd.concat([future_tech, future_furniture, future_office])
future_df.sort_index(inplace=True)

""""Data Setup"""
#train
known_features = SuperStoreCat.iloc[:,1:11]
known_sales = SuperStoreCat.iloc[:,0]
#test
future_features = future_df.iloc[:,1:11]

future_features['Category'] = future_features['Category'].astype("category")

"""Model"""
#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 150,   ##~150 stop in cross validation...no validation set so will overfit if allowed
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1,
                       enable_categorical=True,  ##allows categorical data
                       tree_method = 'hist' ##attempt to allow cat. data
                       )


"""train model"""
reg.fit(known_features,known_sales,
        eval_set = [(known_features,known_sales)],
        verbose = 10 #get readouts as fitting
        )

"""Predict sales"""
futures_sales = reg.predict(future_features)

"""Map future sales onto datset"""
predictions = np.array(futures_sales).tolist()
future_df['Sales'] = predictions

"""Comparison of whole set"""
##move index to column
future_df['Order Date'] = future_df.index
SuperStoreCat['Order Date'] = SuperStoreCat.index

plt.figure()
sns.lineplot(x=SuperStoreCat[SuperStoreCat['Category']=='Technology'].index,
         y=SuperStoreCat[SuperStoreCat['Category']=='Technology'].Sales,
         label= 'Technology Sales',
         color = 'Green')
sns.lineplot(x=future_df[future_df['Category']=='Technology'].index,
         y=future_df[future_df['Category']=='Technology'].Sales,
         label= 'Technology Forecast',
         color = 'Red')

sns.lineplot(x=SuperStoreCat[SuperStoreCat['Category']=='Furniture'].index,
         y=SuperStoreCat[SuperStoreCat['Category']=='Furniture'].Sales,
         label= 'Furniture Sales',
         color = 'Blue')
sns.lineplot(x=future_df[future_df['Category']=='Furniture'].index,
         y=future_df[future_df['Category']=='Furniture'].Sales,
         label= 'Furniture Forecast',
         color = 'Orange')

sns.lineplot(x=SuperStoreCat[SuperStoreCat['Category']=='Office Supplies'].index,
         y=SuperStoreCat[SuperStoreCat['Category']=='Office Supplies'].Sales,
         label= 'Office Supplies Sales',
         color = 'Black')
sns.lineplot(x=future_df[future_df['Category']=='Office Supplies'].index,
         y=future_df[future_df['Category']=='Office Supplies'].Sales,
         label= 'Office Supplies Forecast',
         color = 'Brown')

plt.title('Sales with forecast for the next year')
plt.legend()
plt.show()

evals_result_futurecat = reg.evals_result()
print(evals_result_futurecat["validation_0"]["rmse"][-1])
#1064.85

SuperStoreCat['Prediction'] = 'N'
future_df['Prediction'] = 'Y'
SuperStoreCatVis= pd.concat([SuperStoreCat,future_df])

SuperStoreCatVis.to_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreCatv1.csv')

"""Predict next three years"""

"""Forecast future"""
#create empty future cells.
print(indices[-1])  #max date

date_range = pd.date_range('2019-01-01 00:00:00', '2021-12-30 00:00:00',freq='D')

threeyr_data = pd.DataFrame({'Order Date':date_range})
threeyr_data['Sales'] = ''
threeyr_data['Category'] = 'Technology'
threeyr_data.set_index('Order Date',drop = True, inplace = True)
threeyr_data['Year'] = threeyr_data.index.year
threeyr_data['Month'] = threeyr_data.index.month
threeyr_data['Day of Year'] = threeyr_data.index.dayofyear
threeyr_data['Day of Month'] = threeyr_data.index.day
threeyr_data['Day of Week'] = threeyr_data.index.dayofweek
threeyr_data['Week of Year'] = threeyr_data.index.isocalendar().week.astype(int)

threeyr_tech = threeyr_data.copy(deep=True)

threeyr_data = pd.DataFrame({'Order Date':date_range})
threeyr_data['Sales'] = ''
threeyr_data['Category'] = 'Furniture'
threeyr_data.set_index('Order Date',drop = True, inplace = True)
threeyr_data['Year'] = threeyr_data.index.year
threeyr_data['Month'] = threeyr_data.index.month
threeyr_data['Day of Year'] = threeyr_data.index.dayofyear
threeyr_data['Day of Month'] = threeyr_data.index.day
threeyr_data['Day of Week'] = threeyr_data.index.dayofweek
threeyr_data['Week of Year'] = threeyr_data.index.isocalendar().week.astype(int)

threeyr_furniture = threeyr_data.copy(deep=True)

threeyr_data = pd.DataFrame({'Order Date':date_range})
threeyr_data['Sales'] = ''
threeyr_data['Category'] = 'Office Supplies'
threeyr_data.set_index('Order Date',drop = True, inplace = True)
threeyr_data['Year'] = threeyr_data.index.year
threeyr_data['Month'] = threeyr_data.index.month
threeyr_data['Day of Year'] = threeyr_data.index.dayofyear
threeyr_data['Day of Month'] = threeyr_data.index.day
threeyr_data['Day of Week'] = threeyr_data.index.dayofweek
threeyr_data['Week of Year'] = threeyr_data.index.isocalendar().week.astype(int)

threeyr_office = threeyr_data.copy(deep=True)

"""Create Lag features"""
#Separate the three and then concat df.
tech_target_map = SuperStoreCat[SuperStoreCat['Category']=='Technology'].Sales.to_dict()
furniture_target_map = SuperStoreCat[SuperStoreCat['Category']=='Furniture'].Sales.to_dict()
office_target_map = SuperStoreCat[SuperStoreCat['Category']=='Office Supplies'].Sales.to_dict()

threeyr_tech['Lag1'] = (threeyr_tech.index - pd.Timedelta('364 days')).map(tech_target_map)
threeyr_tech['Lag2'] = (threeyr_tech.index - pd.Timedelta('728 days')).map(tech_target_map)
threeyr_tech['Lag3'] = (threeyr_tech.index - pd.Timedelta('1092 days')).map(tech_target_map)

threeyr_furniture['Lag1'] = (threeyr_furniture.index - pd.Timedelta('364 days')).map(furniture_target_map)
threeyr_furniture['Lag2'] = (threeyr_furniture.index - pd.Timedelta('728 days')).map(furniture_target_map)
threeyr_furniture['Lag3'] = (threeyr_furniture.index - pd.Timedelta('1092 days')).map(furniture_target_map)

threeyr_office['Lag1'] = (threeyr_office.index - pd.Timedelta('364 days')).map(office_target_map)
threeyr_office['Lag2'] = (threeyr_office.index - pd.Timedelta('728 days')).map(office_target_map)
threeyr_office['Lag3'] = (threeyr_office.index - pd.Timedelta('1092 days')).map(office_target_map)

threeyr_df = pd.concat([threeyr_tech, threeyr_furniture, threeyr_office])
threeyr_df.sort_index(inplace=True)

""""Data Setup"""
#train
known_features = SuperStoreCat.iloc[:,1:11]
known_sales = SuperStoreCat.iloc[:,0]
#test
future_features = threeyr_df.iloc[:,1:11]

future_features['Category'] = future_features['Category'].astype("category")

"""Model"""
#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 150,   ##~150 stop in cross validation...no validation set so will overfit if allowed
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1,
                       enable_categorical=True,  ##allows categorical data
                       tree_method = 'hist' ##attempt to allow cat. data
                       )


"""train model"""
reg.fit(known_features,known_sales,
        eval_set = [(known_features,known_sales)],
        verbose = 10 #get readouts as fitting
        )

"""Predict sales"""
futures_sales = reg.predict(future_features)

"""Map future sales onto datset"""
predictions = np.array(futures_sales).tolist()
threeyr_df['Sales'] = predictions

evals_result_3yr = reg.evals_result()
print(evals_result_3yr["validation_0"]["rmse"][-1])
#1064.85 (Model is the same....)

SuperStoreCat['Prediction'] = 'N'
threeyr_df['Prediction'] = 'Y'
#SuperStoreCat.drop(columns='Order Date',inplace=True)
threeyrVis= pd.concat([SuperStoreCat,threeyr_df])

threeyrVis.to_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreCatv3yr.csv')

pickle.dump(reg, open(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreCatModelFuture.sav','wb'))

"""Issue with this type of model: Predicts sales for every day, when in real data, that is not the case...the index skips days.
Grouping over week might alleviate this issue."""
"""Over time the seasonality seemingly is lost"""

##################################################################################################################
##################################################################################################################
##Can we break down into product within categories?
##Sales per sub-cat are not daily so it is better to group by week. Lag by week too.

SuperStoreProd = pd.read_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\train.csv')

SuperStoreProd.drop(columns='Postal Code',inplace=True)

"""Outliers in tech?"""
#Not necessarily and outlier, BUT the only one they sold in the product ID. 
#Clearly very expensive and the only one they have ever sold. Unlikely to sell again so removing might make sense.
#Not representative of true forecast.
#inex 2697
print(SuperStoreProd.iloc[2697,:])
SuperStoreProd.drop([2697],inplace=True)

"""Convert Date fields to datetime objects. Need WW-YY to group by"""
SuperStoreProd['YY_WW'] = pd.to_datetime(SuperStoreProd['Order Date']).dt.strftime('%Y-%W')
SuperStoreProd['Week of Year'] = pd.to_datetime(SuperStoreProd['Order Date']).dt.strftime('%W').astype(int)
SuperStoreProd['Order Date'] = pd.to_datetime(SuperStoreProd['Order Date'])

"""ID's are favoured over names, reduces mining complexity"""
SuperStoreProd.drop(columns=['Customer Name','Product Name'],inplace=True)
"""No information gained by Country...only 1 value"""
SuperStoreProd.drop(columns='Country',inplace=True)
"""RowID not valid"""
SuperStoreProd.drop(columns='Row ID',inplace=True)
"""Order ID is always a distinct value...no information gained"""
SuperStoreProd.drop(columns='Order ID',inplace=True)
##Round 3 we drop all except category, sub_cat
SuperStoreProd.drop(columns=['Customer ID','Segment','City','State','Region','Product ID','Ship Mode','Ship Date'],inplace=True)

"""Reorder by order date"""
SuperStoreProd.sort_values(by='Order Date',inplace=True)
print(SuperStoreProd.info())

"""Generate weeks for lag features"""
SuperStoreProd.set_index('Order Date',drop = True,inplace = True)
SuperStoreProd['YY_WW_1'] = pd.to_datetime(SuperStoreProd.index - pd.Timedelta('364 days')).strftime('%Y-%W')
SuperStoreProd['YY_WW_2'] = pd.to_datetime(SuperStoreProd.index - pd.Timedelta('728 days')).strftime('%Y-%W')
SuperStoreProd['YY_WW_3'] = pd.to_datetime(SuperStoreProd.index - pd.Timedelta('1092 days')).strftime('%Y-%W')

"""Year and Month Features"""
#SuperStoreProd.set_index('Order Date',drop = True,inplace = True)
SuperStoreProd['Year'] = SuperStoreProd.index.year
SuperStoreProd['Month'] = SuperStoreProd.index.month

"""In order to forecast we need to decide the level of aggregation..."""
SuperStoreProd = SuperStoreProd.groupby(by=['YY_WW','Category','Sub-Category','Week of Year','YY_WW_1','YY_WW_2','YY_WW_3','Year','Month'],as_index=False).sum()

"""data type"""
SuperStoreProd['Category'] = SuperStoreProd['Category'].astype("category")
SuperStoreProd['Sub-Category'] = SuperStoreProd['Sub-Category'].astype("category")

"""Change index"""
SuperStoreProd.set_index('YY_WW',drop = True,inplace = True)
SuperStoreProd.sort_index(inplace=True)

print(SuperStoreProd.info())

"""Create Lag features"""
#Separate the three and then concat df.
#TARGET MAP
tech_acc_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Accessories')].Sales.to_dict()
tech_phone_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Phones')].Sales.to_dict()
tech_copy_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Copiers')].Sales.to_dict()
tech_mach_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Machines')].Sales.to_dict()

furniture_book_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Bookcases')].Sales.to_dict()
furniture_chair_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Chairs')].Sales.to_dict()
furniture_table_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Tables')].Sales.to_dict()
furniture_furn_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Furnishings')].Sales.to_dict()

office_store_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Storage')].Sales.to_dict()
office_appl_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Appliances')].Sales.to_dict()
office_art_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Art')].Sales.to_dict()
office_bind_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Binders')].Sales.to_dict()
office_supp_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Supplies')].Sales.to_dict()
office_pap_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Paper')].Sales.to_dict()
office_env_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Envelopes')].Sales.to_dict()
office_fast_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Fasteners')].Sales.to_dict()
office_labels_target_map = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Labels')].Sales.to_dict()

#SUBCAT DF
tech_acc_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Accessories')].copy(deep=True)
tech_phone_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Phones')].copy(deep=True)
tech_copy_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Copiers')].copy(deep=True)
tech_mach_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Technology') & (SuperStoreProd['Sub-Category']=='Machines')].copy(deep=True)

furniture_book_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Bookcases')].copy(deep=True)
furniture_chair_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Chairs')].copy(deep=True)
furniture_table_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Tables')].copy(deep=True)
furniture_furn_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Furniture') & (SuperStoreProd['Sub-Category']=='Furnishings')].copy(deep=True)

office_store_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Storage')].copy(deep=True)
office_appl_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Appliances')].copy(deep=True)
office_art_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Art')].copy(deep=True)
office_bind_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Binders')].copy(deep=True)
office_supp_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Supplies')].copy(deep=True)
office_pap_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Paper')].copy(deep=True)
office_env_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Envelopes')].copy(deep=True)
office_fast_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Fasteners')].copy(deep=True)
office_labels_target_df = SuperStoreProd[(SuperStoreProd['Category']=='Office Supplies') & (SuperStoreProd['Sub-Category']=='Labels')].copy(deep=True)

#LAGS
tech_acc_target_df['Lag1'] = tech_acc_target_df.YY_WW_1.map(tech_acc_target_map)
tech_acc_target_df['Lag2'] = tech_acc_target_df.YY_WW_2.map(tech_acc_target_map)
tech_acc_target_df['Lag3'] = tech_acc_target_df.YY_WW_3.map(tech_acc_target_map)

tech_phone_target_df['Lag1'] = tech_phone_target_df.YY_WW_1.map(tech_acc_target_map)
tech_phone_target_df['Lag2'] = tech_phone_target_df.YY_WW_2.map(tech_acc_target_map)
tech_phone_target_df['Lag3'] = tech_phone_target_df.YY_WW_3.map(tech_acc_target_map)

tech_copy_target_df['Lag1'] = tech_copy_target_df.YY_WW_1.map(tech_copy_target_map)
tech_copy_target_df['Lag2'] = tech_copy_target_df.YY_WW_2.map(tech_copy_target_map)
tech_copy_target_df['Lag3'] = tech_copy_target_df.YY_WW_3.map(tech_copy_target_map)

tech_mach_target_df['Lag1'] = tech_mach_target_df.YY_WW_1.map(tech_mach_target_map)
tech_mach_target_df['Lag2'] = tech_mach_target_df.YY_WW_2.map(tech_mach_target_map)
tech_mach_target_df['Lag3'] = tech_mach_target_df.YY_WW_3.map(tech_mach_target_map)

furniture_book_target_df['Lag1'] = furniture_book_target_df.YY_WW_1.map(furniture_book_target_map)
furniture_book_target_df['Lag2'] = furniture_book_target_df.YY_WW_2.map(furniture_book_target_map)
furniture_book_target_df['Lag3'] = furniture_book_target_df.YY_WW_3.map(furniture_book_target_map)

furniture_chair_target_df['Lag1'] = furniture_chair_target_df.YY_WW_1.map(furniture_chair_target_map)
furniture_chair_target_df['Lag2'] = furniture_chair_target_df.YY_WW_2.map(furniture_chair_target_map)
furniture_chair_target_df['Lag3'] = furniture_chair_target_df.YY_WW_3.map(furniture_chair_target_map)

furniture_table_target_df['Lag1'] = furniture_table_target_df.YY_WW_1.map(furniture_table_target_map)
furniture_table_target_df['Lag2'] = furniture_table_target_df.YY_WW_2.map(furniture_table_target_map)
furniture_table_target_df['Lag3'] = furniture_table_target_df.YY_WW_3.map(furniture_table_target_map)

furniture_furn_target_df['Lag1'] = furniture_furn_target_df.YY_WW_1.map(furniture_furn_target_map)
furniture_furn_target_df['Lag2'] = furniture_furn_target_df.YY_WW_2.map(furniture_furn_target_map)
furniture_furn_target_df['Lag3'] = furniture_furn_target_df.YY_WW_3.map(furniture_furn_target_map)

office_store_target_df['Lag1'] = office_store_target_df.YY_WW_1.map(office_store_target_map)
office_store_target_df['Lag2'] = office_store_target_df.YY_WW_2.map(office_store_target_map)
office_store_target_df['Lag3'] = office_store_target_df.YY_WW_3.map(office_store_target_map)

office_appl_target_df['Lag1'] = office_appl_target_df.YY_WW_1.map(office_appl_target_map)
office_appl_target_df['Lag2'] = office_appl_target_df.YY_WW_2.map(office_appl_target_map)
office_appl_target_df['Lag3'] = office_appl_target_df.YY_WW_3.map(office_appl_target_map)

office_art_target_df['Lag1'] = office_art_target_df.YY_WW_1.map(office_art_target_map)
office_art_target_df['Lag2'] = office_art_target_df.YY_WW_2.map(office_art_target_map)
office_art_target_df['Lag3'] = office_art_target_df.YY_WW_3.map(office_art_target_map)

office_bind_target_df['Lag1'] = office_bind_target_df.YY_WW_1.map(office_bind_target_map)
office_bind_target_df['Lag2'] = office_bind_target_df.YY_WW_2.map(office_bind_target_map)
office_bind_target_df['Lag3'] = office_bind_target_df.YY_WW_3.map(office_bind_target_map)

office_supp_target_df['Lag1'] = office_supp_target_df.YY_WW_1.map(office_supp_target_map)
office_supp_target_df['Lag2'] = office_supp_target_df.YY_WW_2.map(office_supp_target_map)
office_supp_target_df['Lag3'] = office_supp_target_df.YY_WW_3.map(office_supp_target_map)

office_pap_target_df['Lag1'] = office_pap_target_df.YY_WW_1.map(office_pap_target_map)
office_pap_target_df['Lag2'] = office_pap_target_df.YY_WW_2.map(office_pap_target_map)
office_pap_target_df['Lag3'] = office_pap_target_df.YY_WW_3.map(office_pap_target_map)

office_env_target_df['Lag1'] = office_env_target_df.YY_WW_1.map(office_env_target_map)
office_env_target_df['Lag2'] = office_env_target_df.YY_WW_2.map(office_env_target_map)
office_env_target_df['Lag3'] = office_env_target_df.YY_WW_3.map(office_env_target_map)

office_fast_target_df['Lag1'] = office_fast_target_df.YY_WW_1.map(office_fast_target_map)
office_fast_target_df['Lag2'] = office_fast_target_df.YY_WW_2.map(office_fast_target_map)
office_fast_target_df['Lag3'] = office_fast_target_df.YY_WW_3.map(office_fast_target_map)

office_labels_target_df['Lag1'] = office_labels_target_df.YY_WW_1.map(office_labels_target_map)
office_labels_target_df['Lag2'] = office_labels_target_df.YY_WW_2.map(office_labels_target_map)
office_labels_target_df['Lag3'] = office_labels_target_df.YY_WW_3.map(office_labels_target_map)

#Concat and re-sort
SuperStoreProd = pd.concat([tech_acc_target_df,
                            tech_phone_target_df,
                            tech_copy_target_df,
                            tech_mach_target_df,
                            furniture_book_target_df,
                            furniture_chair_target_df,
                            furniture_table_target_df,
                            furniture_furn_target_df,
                            office_store_target_df,
                            office_appl_target_df,
                            office_art_target_df,
                            office_bind_target_df,
                            office_supp_target_df,
                            office_pap_target_df,
                            office_env_target_df,
                            office_fast_target_df,
                            office_labels_target_df])

SuperStoreProd.sort_index(inplace=True)

"""Drop map fields"""
SuperStoreProd.drop(columns=['YY_WW_1','YY_WW_2','YY_WW_3'],inplace=True)

"""Data Prep complete."""
"""Rearrange columns to have sales at front"""
print(SuperStoreProd.columns)

SuperStoreProd = SuperStoreProd[['Sales', 'Category', 'Sub-Category', 'Year', 'Month','Week of Year', 'Lag1', 'Lag2', 'Lag3']]

"""Cross Validation"""
#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 500,   ##Now we have validation set, can utilise early stopping.
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1,
                       enable_categorical=True,  ##allows categorical data
                       tree_method = 'hist' ##attempt to allow cat. data
                       )


scores = []
folds = []
trees = []
fold = 1


for train_index, validation_index in timeseriessplit.split(SuperStoreProd):
    tr_data = SuperStoreProd.iloc[train_index]
    te_data = SuperStoreProd.iloc[validation_index]
    #train
    tr_features = tr_data.iloc[:,1:9]
    tr_target = tr_data.iloc[:,0]
    #test
    te_features = te_data.iloc[:,1:9]
    te_target = te_data.iloc[:,0]
    """train model"""
    reg.fit(tr_features,tr_target,
            eval_set = [(tr_features,tr_target),(te_features,te_target)],
            verbose = 10 #get readouts as fitting
            )
    sales_pred = reg.predict(te_features)
    evals_result = reg.evals_result()
    scores.append(evals_result["validation_1"]["rmse"][-1])
    trees.append(reg.best_ntree_limit)
    folds.append(fold)
    fold = fold+1
    
print(scores)
print(np.mean(scores))
##1246
#trees around 150

"""Most important features"""
print(reg.feature_names_in_, "importance:", reg.feature_importances_)

print(SuperStoreProd.Sales.mean())

sales_pred = reg.predict(te_features)

"""Compare test to forecast"""

evals_result_futurecat = reg.evals_result()
print(evals_result_futurecat["validation_1"]["rmse"][-1])

te_data.to_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreProd_test_results.csv')
pickle.dump(reg, open(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreSCModel.sav','wb'))

################################################################################

"""Forecast future"""
#create empty future cells.
print(indices[-1])  #max date

future_prod = pd.DataFrame()

date_range = pd.date_range('2018-12-31 00:00:00', '2019-12-31 00:00:00',freq='W')

cat_subcat = {'Technology': ['Phones','Machines','Accessories','Copiers'],
              'Office Supplies': ['Labels','Binders','Fasteners','Paper','Storage','Art','Appliances','Envelopes','Supplies'],
              'Furniture': ['Bookcases','Chairs','Tables','Furnishings']
              }

for x in cat_subcat:
    for y in cat_subcat[x]:
        print("x: ",x, '\n','y: ', y)
        print("future_{}".format(y))
              
for x in cat_subcat:
    for y in cat_subcat[x]:
        future_data = pd.DataFrame({'Order Date':date_range})
        future_data['Sales'] = ''
        future_data['Category'] = x
        future_data['Sub-Category'] = y
        future_data['Week of Year'] = pd.to_datetime(future_data['Order Date']).dt.strftime('%W').astype(int)
        future_data['YY_WW'] = pd.to_datetime(future_data['Order Date']).dt.strftime('%Y-%W')
        future_data.set_index('Order Date',drop = True, inplace = True)
        future_data['Year'] = future_data.index.year
        future_data['Month'] = future_data.index.month
        future_data['YY_WW_1'] = pd.to_datetime(future_data.index - pd.Timedelta('364 days')).strftime('%Y-%W')
        future_data['YY_WW_2'] = pd.to_datetime(future_data.index - pd.Timedelta('728 days')).strftime('%Y-%W')
        future_data['YY_WW_3'] = pd.to_datetime(future_data.index - pd.Timedelta('1092 days')).strftime('%Y-%W')
        future_prod = pd.concat([future_prod,future_data])


#SUBCAT DF
future_acc_df = future_prod[(future_prod['Category']=='Technology') & (future_prod['Sub-Category']=='Accessories')].copy(deep=True)
future_phone_df = future_prod[(future_prod['Category']=='Technology') & (future_prod['Sub-Category']=='Phones')].copy(deep=True)
future_copy_df = future_prod[(future_prod['Category']=='Technology') & (future_prod['Sub-Category']=='Copiers')].copy(deep=True)
future_mach_df = future_prod[(future_prod['Category']=='Technology') & (future_prod['Sub-Category']=='Machines')].copy(deep=True)

future_book_df = future_prod[(future_prod['Category']=='Furniture') & (future_prod['Sub-Category']=='Bookcases')].copy(deep=True)
future_chair_df = future_prod[(future_prod['Category']=='Furniture') & (future_prod['Sub-Category']=='Chairs')].copy(deep=True)
future_table_df = future_prod[(future_prod['Category']=='Furniture') & (future_prod['Sub-Category']=='Tables')].copy(deep=True)
futures_furnishings_df = future_prod[(future_prod['Category']=='Furniture') & (future_prod['Sub-Category']=='Furnishings')].copy(deep=True)

future_storage_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Storage')].copy(deep=True)
future_appliances_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Appliances')].copy(deep=True)
futures_art_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Art')].copy(deep=True)
future_binders_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Binders')].copy(deep=True)
future_supplies_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Supplies')].copy(deep=True)
future_paper_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Paper')].copy(deep=True)
future_envelopes_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Envelopes')].copy(deep=True)
future_fasteners_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Fasteners')].copy(deep=True)
future_labels_df = future_prod[(future_prod['Category']=='Office Supplies') & (future_prod['Sub-Category']=='Labels')].copy(deep=True)

#LAGS
future_acc_df['Lag1'] = future_acc_df.YY_WW_1.map(tech_acc_target_map)
future_acc_df['Lag2'] = future_acc_df.YY_WW_2.map(tech_acc_target_map)
future_acc_df['Lag3'] = future_acc_df.YY_WW_3.map(tech_acc_target_map)

future_phone_df['Lag1'] = future_phone_df.YY_WW_1.map(tech_acc_target_map)
future_phone_df['Lag2'] = future_phone_df.YY_WW_2.map(tech_acc_target_map)
future_phone_df['Lag3'] = future_phone_df.YY_WW_3.map(tech_acc_target_map)

future_copy_df['Lag1'] = future_copy_df.YY_WW_1.map(tech_copy_target_map)
future_copy_df['Lag2'] = future_copy_df.YY_WW_2.map(tech_copy_target_map)
future_copy_df['Lag3'] = future_copy_df.YY_WW_3.map(tech_copy_target_map)

future_mach_df['Lag1'] = future_mach_df.YY_WW_1.map(tech_mach_target_map)
future_mach_df['Lag2'] = future_mach_df.YY_WW_2.map(tech_mach_target_map)
future_mach_df['Lag3'] = future_mach_df.YY_WW_3.map(tech_mach_target_map)

future_book_df['Lag1'] = future_book_df.YY_WW_1.map(furniture_book_target_map)
future_book_df['Lag2'] = future_book_df.YY_WW_2.map(furniture_book_target_map)
future_book_df['Lag3'] = future_book_df.YY_WW_3.map(furniture_book_target_map)

future_chair_df['Lag1'] = future_chair_df.YY_WW_1.map(furniture_chair_target_map)
future_chair_df['Lag2'] = future_chair_df.YY_WW_2.map(furniture_chair_target_map)
future_chair_df['Lag3'] = future_chair_df.YY_WW_3.map(furniture_chair_target_map)

future_table_df['Lag1'] = future_table_df.YY_WW_1.map(furniture_table_target_map)
future_table_df['Lag2'] = future_table_df.YY_WW_2.map(furniture_table_target_map)
future_table_df['Lag3'] = future_table_df.YY_WW_3.map(furniture_table_target_map)

futures_furnishings_df['Lag1'] = futures_furnishings_df.YY_WW_1.map(furniture_furn_target_map)
futures_furnishings_df['Lag2'] = futures_furnishings_df.YY_WW_2.map(furniture_furn_target_map)
futures_furnishings_df['Lag3'] = futures_furnishings_df.YY_WW_3.map(furniture_furn_target_map)

future_storage_df['Lag1'] = future_storage_df.YY_WW_1.map(office_store_target_map)
future_storage_df['Lag2'] = future_storage_df.YY_WW_2.map(office_store_target_map)
future_storage_df['Lag3'] = future_storage_df.YY_WW_3.map(office_store_target_map)

future_appliances_df['Lag1'] = future_appliances_df.YY_WW_1.map(office_appl_target_map)
future_appliances_df['Lag2'] = future_appliances_df.YY_WW_2.map(office_appl_target_map)
future_appliances_df['Lag3'] = future_appliances_df.YY_WW_3.map(office_appl_target_map)

futures_art_df['Lag1'] = futures_art_df.YY_WW_1.map(office_art_target_map)
futures_art_df['Lag2'] = futures_art_df.YY_WW_2.map(office_art_target_map)
futures_art_df['Lag3'] = futures_art_df.YY_WW_3.map(office_art_target_map)

future_binders_df['Lag1'] = future_binders_df.YY_WW_1.map(office_bind_target_map)
future_binders_df['Lag2'] = future_binders_df.YY_WW_2.map(office_bind_target_map)
future_binders_df['Lag3'] = future_binders_df.YY_WW_3.map(office_bind_target_map)

future_supplies_df['Lag1'] = future_supplies_df.YY_WW_1.map(office_supp_target_map)
future_supplies_df['Lag2'] = future_supplies_df.YY_WW_2.map(office_supp_target_map)
future_supplies_df['Lag3'] = future_supplies_df.YY_WW_3.map(office_supp_target_map)

future_paper_df['Lag1'] = future_paper_df.YY_WW_1.map(office_pap_target_map)
future_paper_df['Lag2'] = future_paper_df.YY_WW_2.map(office_pap_target_map)
future_paper_df['Lag3'] = future_paper_df.YY_WW_3.map(office_pap_target_map)

future_envelopes_df['Lag1'] = future_envelopes_df.YY_WW_1.map(office_env_target_map)
future_envelopes_df['Lag2'] = future_envelopes_df.YY_WW_2.map(office_env_target_map)
future_envelopes_df['Lag3'] = future_envelopes_df.YY_WW_3.map(office_env_target_map)

future_fasteners_df['Lag1'] = future_fasteners_df.YY_WW_1.map(office_fast_target_map)
future_fasteners_df['Lag2'] = future_fasteners_df.YY_WW_2.map(office_fast_target_map)
future_fasteners_df['Lag3'] = future_fasteners_df.YY_WW_3.map(office_fast_target_map)

future_labels_df['Lag1'] = future_labels_df.YY_WW_1.map(office_labels_target_map)
future_labels_df['Lag2'] = future_labels_df.YY_WW_2.map(office_labels_target_map)
future_labels_df['Lag3'] = future_labels_df.YY_WW_3.map(office_labels_target_map)

#Concat and re-sort
future_prod = pd.concat([future_acc_df,
                         future_phone_df,
                         future_copy_df,
                         future_mach_df,
                         future_book_df,
                         future_chair_df,
                         future_table_df,
                         futures_furnishings_df,
                         future_storage_df,
                         future_appliances_df,
                         futures_art_df,
                         future_binders_df,
                         future_supplies_df,
                         future_paper_df,
                         future_envelopes_df,
                         future_fasteners_df,
                         future_labels_df])

future_prod.sort_index(inplace=True)

"""Drop map fields"""
future_prod.drop(columns=['YY_WW_1','YY_WW_2','YY_WW_3'],inplace=True)

"""Data Prep complete."""
"""Change index"""
future_prod.set_index('YY_WW',drop = True,inplace = True)
future_prod.sort_index(inplace=True)

"""Rearrange columns to have sales at front"""
print(future_prod.columns)
future_prod = future_prod[['Sales', 'Category', 'Sub-Category', 'Year', 'Month','Week of Year', 'Lag1', 'Lag2', 'Lag3']]

print(future_prod.info())

future_prod['Category'] = future_prod['Category'].astype("category")
future_prod['Sub-Category'] = future_prod['Sub-Category'].astype("category")

""""Data Setup"""
#train
known_features = SuperStoreProd.iloc[:,1:9]
known_sales = SuperStoreProd.iloc[:,0]
#test
future_features = future_prod.iloc[:,1:9]

"""Model"""
#initialise
reg = xgb.XGBRegressor(base_score = 0.5,
                       booster = 'gbtree',
                       verbosity = 1,
                       learning_rate = 0.01,  ##learning rate imppacts how quickly we get to the endpoint of learning.
                       max_depth = 6,
                       objective = 'reg:squarederror',
                       n_estimators = 150,   ##~150 stop in cross validation...no validation set so will overfit if allowed
                       early_stopping_rounds = 7, ##if no RMSE improvement stop after x rounds
                       seed = 1,
                       enable_categorical=True,  ##allows categorical data
                       tree_method = 'hist' ##attempt to allow cat. data
                       )


"""train model"""
reg.fit(known_features,known_sales,
        eval_set = [(known_features,known_sales)],
        verbose = 10 #get readouts as fitting
        )

"""Predict sales"""
futures_sales = reg.predict(future_features)

"""Map future sales onto datset"""
predictions = np.array(futures_sales).tolist()
future_prod['Sales'] = predictions

evals_result_prod_forecast = reg.evals_result()
print(evals_result_prod_forecast["validation_0"]["rmse"][-1])
#963.47

SuperStoreProd['Prediction'] = 'N'
future_prod['Prediction'] = 'Y'
SuperStoreProdVis= pd.concat([SuperStoreProd,future_prod])

SuperStoreProdVis.to_csv(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreProdForecast.csv')

pickle.dump(reg, open(r'C:\Users\jordan.ottewill\OneDrive - Wesco Aircraft Hardware Corporation\Apprenticeship\DMML\SuperStoreSCModelFuture.sav','wb'))

