#!/usr/bin/env python
# coding: utf-8

# # Zillow Prize: Zillow’s Home Value Prediction (Zestimate)

# Zillow’s Zestimate home valuation has shaken up the U.S. real estate industry since first released 11 years ago.
# 
# A home is often the largest and most expensive purchase a person makes in his or her lifetime. Ensuring homeowners have a trusted way to monitor this asset is incredibly important. The Zestimate was created to give consumers as much information as possible about homes and the housing market, marking the first time consumers had access to this type of home value information at no cost.
# 
# “Zestimates” are estimated home values based on 7.5 million statistical and machine learning models that analyze hundreds of data points on each property. And, by continually improving the median margin of error (from 14% at the onset to 5% today), Zillow has since become established as one of the largest, most trusted marketplaces for real estate information in the U.S. and a leading example of impactful machine learning.
# 
# #### Zillow is asking you to predict the log-error between their Zestimate and the actual sale price, given all the features of a home.

# # overview
# 
# ### Downloading the data
# - Loading it into a dataframe
# - Describing dataset
# 
# 
# ### Proccessing and Feature Engineering
# - Duplicate values
# - Merging of property and training dataset
# - Missing Valuse above 35% droped
# - Date column
# 
# 
# ### Exploratory Data Analysis and Visualisation
# - Date
# - Parcel Location
# - Target Variable
# - correlation Heatmap
# 
# ### Identifying Input and Target Columns
# 
# ### Categorical and Numeric Columns
# - Encoded categorical columns
# - Imputing missing numerical columns
# - Scale numeric values
# 
# ### Splitting the data for training
# - Training data(X_train)
# - Validation data(X_val)
# 
# ## Training and Tuning Different Model
# - Random Forest Regression
# - XGBRegressor
# - Gradient Boosting Regression
# 
# ### Training Final Model
# - Gradient Boost Regression
# 
# ### Saving The Model
# - Using joblib
# 
# ### Test Ptrdiction
# - conclution
# 
# ### Conclusion
# - Summary
# - Downside
# - Limitations

# Import libraries to be used

# In[1]:


import os
import opendatasets as od
import pandas as pd
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 120)


# Downloading the dataset from kaggle

# In[5]:


od.download('https://www.kaggle.com/c/zillow-prize-1')


# In[2]:


os.listdir('.\zillow-prize-1')


# In[4]:


prop_raw = pd.read_csv('.\zillow-prize-1/properties_2016.csv')
train_raw = pd.read_csv('.\zillow-prize-1/train_2016_v2.csv')


# In[5]:


prop_raw.head(6)


# In[6]:


train_raw.head(6)


# ### Details

# 

# In[7]:


def get_unique(name,data_series):
    print("{} has total {} records and {} are unique.".format(name,len(data_series),len(data_series.unique())))


# In[8]:


get_unique('train_raw_df', train_raw['parcelid'])


# In[9]:


get_unique('property_raw_df', prop_raw['parcelid'])


# ### Preprocessing and Feature Engineering
# Let's take a look at the available columns, and figure out if we can create new columns or apply any useful transformations.

# ### Duplicate Values
# Train data has some duplicate values Let's analyse those duplicate values to get better understanding.

# In[10]:


duplicate_df = train_raw[train_raw.duplicated(["parcelid"],keep=False)]
print("All Duplicate Rows based on all columns are :")
pd.DataFrame(duplicate_df.head(10))


# In[11]:


duplicate_df["parcelid"].value_counts()


# In[12]:


duplicate_df.loc[duplicate_df["parcelid"]==11842707]


# Here, we can observe that some houses were sold earlier in the year 2016 and after some month they were sold again in the same year. so we will consider the last selling price of this type of house.

# In[13]:


unique_train_df = train_raw.sort_values("transactiondate").drop_duplicates("parcelid",keep = "last")


# In[14]:


get_unique("unique_train_df",unique_train_df["parcelid"])


# ### Merging two dataset for extensive EDA

# In[15]:


merged_df = pd.merge(prop_raw, unique_train_df, on='parcelid', how='left')


# In[16]:


get_unique('merged_df', merged_df['parcelid'])


# In[17]:


merged_df['logerror'].value_counts().sum()


# ### Missing Values
# Dealing with missing Values

# First of all lets look at the target columns to see if there are missing values

# In[20]:


merged_df['logerror'].isna().sum()


# - It seems there are missing values on the target column which will not be good for training our model
# - Removing all the rows with null values 

# In[21]:


train_df = merged_df[merged_df['logerror'].notna()]


# In[22]:


len(train_df)


# From the description of the dataset on kaggle, it was shown that there are so many column that have a lot of missing values

# Checking for columns with missing values greater than 35%

# In[30]:


def drop_columns(data):
    missing_value_df = pd.DataFrame((data.isnull().sum()/len(data))*100,columns=["missing_value"])
    drop_columns_list = missing_value_df.loc[missing_value_df["missing_value"]>35].index.to_list()
    return drop_columns_list


# In[31]:


drop_col = drop_columns(train_df)


# These are the columns to be droped that have more than 30% missing values

# In[32]:


drop_col


# In[33]:


train_df.drop(columns=drop_col,inplace = True)


# In[34]:


(train_df.isnull().sum()/len(train_df))*100


# we can now see from the data above that there are only columns with nan values below 35%

# In[35]:


train_df.dtypes


# In[ ]:





# ### Date Columnns

# The data set only contains 2016 as the year. i will create a new column for month and day living out the year

# In[37]:


def split_date_df(df):
    df['Date'] = pd.to_datetime(df['transactiondate'])
    df['Month'] = df.Date.dt.month
    df['Day'] = df.Date.dt.day
    df['WeekOfYear'] = df.Date.dt.isocalendar().week


# In[38]:


split_date_df(train_df)


# In[39]:


train_df


# Since transactiondata and Data column are now the same and will not be needed, i will drop them

# In[40]:


train_df.drop(columns=['transactiondate', 'Date'],inplace = True)


# In[41]:


train_df


# ### Exploratory Data Analysis and Visualization

# I will be using some python Libraries for the visualization to see the relation between other columns and the customers

# In[90]:


import plotly.express as px
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[91]:


sns.set_style('darkgrid')
matplotlib.rcParams['font.size'] = 14
matplotlib.rcParams['figure.figsize'] = (10, 6)
matplotlib.rcParams['figure.facecolor'] = '#00000000'


# In[241]:


train_df


# ##### Dates

# In[ ]:





# In[244]:


sns.barplot(x='Month', y='logerror', data=train_df);


# December is has the highest number of logerror

# ##### Parcel Locations
# Let's explore location/region related variables! Variables written below are used to plot regions.
# 
# 1 longitude & latitude
# 2 regions: regionidcounty, regionidcity, regionidzip

# Overall region looks like below! According to Kaggle description, data from Los Angeles, Orange, Ventura are included

# In[249]:


# overall region plot
plt.figure(figsize = (8,4))
ax = plt.subplot(1,1,1)
plt.plot(train_df['longitude'], train_df['latitude'], 'o', markersize = 0.2, color = '#004c70');
plt.xlabel('Longitude'); plt.ylabel('Latitude'); plt.title('Locations of parcels', fontsize = 15)

for s in ["top","right","left"]:
    ax.spines[s].set_visible(False)

plt.tight_layout()


# In[ ]:





# #### Target variable: Logerror
# Target variable, Logerror ranges from -5 to 5, but mostly distributed around 0. Plus, distribution plots of year 2016 and 2017 are almost same!

# In[248]:


ax = plt.subplot(1,1,1)
sns.distplot(train_df['logerror'], color = '#004c70')
plt.title('Overall distribution of Logerror', fontsize = 15)

for s in ['top','left','right']:
    ax.spines[s].set_visible(False)
ax.grid(axis='y', linestyle='-', alpha=0.4)
plt.show();


# In[ ]:





# ##### Correlation Heatmap
# Graph below shows correlation heatmap based on correlation coefficients with logerror. Overall coeffs are really small, maximum is 0.04. According to this heatmap, important variables are finishedsquarefeet12, calcuatedfinishedsquarefeet, calculatedbathnbr, bedroomcnt, fullbathcnt, bathroomcnt..etc

# In[253]:


corr = pd.DataFrame(train_df.corr()['logerror'].sort_values(ascending = False)).rename(columns = {'logerror':'correlation'})

plt.figure(figsize = (3,8))
sns.heatmap(corr, annot = True, fmt = '.2f', vmin = -0.05, vmax = 0.05, cmap = 'YlGnBu')
plt.title('Correlation heatmap', fontsize = 15)
plt.show()


# In[ ]:





# ### Input and Target Columns

# In[42]:


train_df.columns


# In[71]:


input_cols = ['bathroomcnt', 'bedroomcnt', 'calculatedbathnbr',
       'calculatedfinishedsquarefeet', 'finishedsquarefeet12', 'fips',
       'fullbathcnt', 'latitude', 'longitude', 'lotsizesquarefeet',
       'propertycountylandusecode', 'propertylandusetypeid',
       'rawcensustractandblock', 'regionidcity', 'regionidcounty',
       'regionidzip', 'roomcnt', 'yearbuilt', 'structuretaxvaluedollarcnt',
       'taxvaluedollarcnt', 'assessmentyear', 'landtaxvaluedollarcnt',
       'taxamount', 'censustractandblock', 'Month', 'Day',
       'WeekOfYear'
]
targets_cols = 'logerror'


# In[46]:


inputs = train_df[input_cols].copy()


# In[72]:


targets = train_df[targets_cols].copy()


# In[47]:


inputs


# In[73]:


targets


# ### Categorical and Numeric Columns

# In[49]:


import numpy as np


# In[50]:


numeric_cols = inputs.select_dtypes(include=np.number).columns.tolist()
categorical_cols = inputs.select_dtypes('object').columns.tolist()


# In[51]:


numeric_cols


# In[52]:


categorical_cols


# we hava only one categorical column

# In[55]:


inputs[categorical_cols].nunique()


# we have about 77 unique values that for a dataset of about 90 thousand rows

# ### Encode Categorical Columns
# Using Onehotencoded

# In[56]:


from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(inputs[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))


# In[57]:


inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])


# In[58]:


inputs


# ### Impute missing numerical data

# In[59]:


inputs[numeric_cols].isna().sum()


# we still have some missing values which is ok to handle rather than droping the column

# In[61]:


from sklearn.impute import SimpleImputer


# In[63]:


imputer = SimpleImputer().fit(inputs[numeric_cols])


# In[64]:


inputs[numeric_cols] = imputer.transform(inputs[numeric_cols])


# ### Scale Numeric Values
# Let's scale numeric values to the 0 to 1 range.

# In[65]:


from sklearn.preprocessing import MinMaxScaler


# In[66]:


scaler = MinMaxScaler().fit(inputs[numeric_cols])
inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])


# In[69]:


inputs[numeric_cols].describe()


# The data is now set for training

# In[74]:


train_inputs = inputs[numeric_cols + encoded_cols]
train_inputs


# ### Spliting of the data sets into train and validation set

# In[70]:


from sklearn.model_selection import train_test_split


# In[80]:


X_train, X_val, train_targets, val_targets = train_test_split(train_inputs, targets, test_size=0.2, random_state=42)


# In[81]:


X_train


# In[82]:


train_targets


# In[83]:


X_val


# In[84]:


val_targets


# 

# ### Random Forest

# In[118]:


from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


# In[86]:


ran_model = RandomForestRegressor(random_state=42, n_jobs=-1)


# In[87]:


ran_model.fit(X_train, train_targets)


# In[88]:


ran_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': ran_model.feature_importances_
}).sort_values('importance', ascending=False)


# In[93]:


plt.title('Random Forest Feature Importance')
sns.barplot(data=ran_importance_df.head(15), x='importance', y='feature');


# We can see that there are so many factors that contributes to the logerror

# In[119]:


train_pred = ran_model.predict(X_train)


# In[123]:


mean_squared_error(train_targets, train_pred, squared=False)


# In[124]:


val_pred = ran_model.predict(X_val)


# In[125]:


mean_squared_error(val_targets, val_pred, squared=False)


# ### Hyperparameter Tuning
# I will be using few parameters such as:
# 
# - max_depth
# - n_estimators
# - max_features

# In[126]:


def params(**params):
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params).fit(X_train, train_targets)
    train_preds, val_preds = model.predict(X_train), model.predict(X_val)
    train_mean_error= mean_squared_error(train_targets, train_preds, squared=False)
    val_mean_error = mean_squared_error(val_targets, val_preds, squared=False)
    return f"train_mean_error is {train_mean_error} and val_mean_error is {val_mean_error}"


# In[127]:


params()


# In[145]:


params(max_depth=3)


# In[136]:


params(n_estimators=50)


# In[185]:


params(min_samples_split=4)


# In[132]:


params(max_features=0.4)


# In[144]:


params(max_features=0.5)


# In[142]:


params(max_depth=5)


# ##### Training the final parameters on the model

# In[187]:


params(max_depth=3, max_features=0.4, n_estimators=70, min_samples_split=5)


# ### XGBRegressor

# In[152]:


from xgboost import XGBRegressor


# In[153]:


xgb_model = XGBRegressor(random_state=42, n_jobs=-1).fit(X_train, train_targets)


# In[154]:


xgb_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)


# In[155]:


plt.title('XGBRegressor Feature Importance')
sns.barplot(data=xgb_importance_df.head(15), x='importance', y='feature');


# In[156]:


def xgboost(**params):
    model = XGBRegressor(random_state=42, n_jobs=-1, **params).fit(X_train, train_targets)
    train_preds, val_preds = model.predict(X_train), model.predict(X_val)
    train_mean_error= mean_squared_error(train_targets, train_preds, squared=False)
    val_mean_error = mean_squared_error(val_targets, val_preds, squared=False)
    return f"train_mean_error is {train_mean_error} and val_mean_error is {val_mean_error}"


# In[157]:


xgboost()


# In[160]:


xgboost(n_estimators=20)


# In[161]:


xgboost(learning_rate=0.1)


# In[165]:


xgboost(learnig_rate=0.3)


# In[162]:


xgboost(n_estimators=50)


# In[163]:


xgboost(max_depth=18)


# #### Final Xgboost parameter model training

# In[166]:


xgboost(n_estimators=20, learning_rate=0.1)


# ### Gradient Boosting

# In[167]:


from sklearn.ensemble import GradientBoostingRegressor


# In[168]:


grad_model = GradientBoostingRegressor(random_state=42).fit(X_train, train_targets)


# In[169]:


grad_importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': grad_model.feature_importances_
}).sort_values('importance', ascending=False)


# In[170]:


plt.title('Gradient Boosting Feature Importance')
sns.barplot(data=grad_importance_df.head(15), x='importance', y='feature');


# In[171]:


def gboost(**params):
    model = GradientBoostingRegressor(random_state=42, **params).fit(X_train, train_targets)
    train_preds, val_preds = model.predict(X_train), model.predict(X_val)
    train_mean_error= mean_squared_error(train_targets, train_preds, squared=False)
    val_mean_error = mean_squared_error(val_targets, val_preds, squared=False)
    return f"train_mean_error is {train_mean_error} and val_mean_error is {val_mean_error}"


# In[172]:


gboost()


# In[176]:


gboost(learning_rate=0.1)


# In[184]:


gboost(n_estimators=15)


# In[183]:


gboost(min_samples_split=4)


# In[179]:


gboost(n_estimators=20)


# In[231]:


gboost(learning_rate=0.1, n_estimators=50, min_samples_split=4)


# In[ ]:





# In[ ]:





# In[ ]:





# ### Train the final model
# The final model will be trained

# In[227]:


model = GradientBoostingRegressor(random_state=42, learning_rate=0.1, n_estimators=50, min_samples_split=4).fit(train_inputs, targets)


# In[228]:


train_predicts = model.predict(train_inputs)


# In[229]:


mean_squared_error(targets, train_predicts, squared=False)


# ### Saving the model

# In[204]:


import joblib


# In[205]:


zillow_prize_rf = {
    'model': model,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': targets_cols,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}


# In[206]:


joblib.dump(zillow_prize_rf, 'zillow_prize_rf.joblib')


# In[ ]:





# ### Testing dataset

# In[233]:


def test_cleaner(prop_raw, test_df):
    ## handling duplicate values
    unique_test_df = test_df.sort_values("transactiondate").drop_duplicates("parcelid",keep = "last")
    merged_df = pd.merge(prop_raw, unique_test_df, on='parcelid', how='left')
    ## date colums
    split_date_df(merged_df)
    ##inputs col
    inputs = merged_df[input_cols].copy()
    ## encoded col
    inputs[encoded_cols] = encoder.transform(inputs[categorical_cols])
    ## missing values
    inputs[numeric_cols] = imputer.transform(inputs[numeric_cols])
    ## scaling numeric col
    inputs[numeric_cols] = scaler.transform(inputs[numeric_cols])
    ## training inputs
    test_inputs = inputs[numeric_cols + encoded_cols]
    return test_inputs


# In[225]:


os.listdir('zillow-prize-1')


# In[232]:


test_raw = pd.read_csv('.\zillow-prize-1/train_2017.csv')
test_prop = pd.read_csv('.\zillow-prize-1/properties_2017.csv')
sample_sub = pd.read_csv('.\zillow-prize-1/sample_submission.csv')


# In[235]:


sample_sub


# In[236]:


test_raw


# ## Conclusion
# The model was trained using Gradient boosting regressor
# ##### - Summary
# The data set was gotten from kaggle 
# - There were duplicate of parselid rows which was removed
# - Some houses were sold and were later resold but the last was put into consederation
# - With 
#  
# 
# ##### Limitations
# - I have a slow system that made me not to use xgboost for train and testing which would have drastically improve the model
# - Slow cpu
# - all downsides and errors that may be found is associated to my slow pc that was why it was igored
# 

# In[237]:


import jovian


# In[254]:


jovian.commit(outputs='zillow_prize_rf.joblib')


# Inspirational Materials
# https://jovian.ai/fidekg123/python-gradient-boosting-machines
# https://jovian.ai/fidekg123/rossmann-stores-customers
# https://www.kaggle.com/yurimhwang/zillow-prize-zillow-s-home-value-prediction-yr?rvi=1
# https://www.kaggle.com/alikashif1994/zillow-scr?rvi=1
# https://jovian.ai/fidekg123/python-random-forests-assignment
# https://www.kaggle.com/noey26/modeling-project?rvi=1
# https://www.kaggle.com/hyewon328/zillow-analysis-with-eda?rvi=1
# https://www.kaggle.com/c/zillow-prize-1?rvi=1

# In[ ]:


jovian.commit()


# In[ ]:




