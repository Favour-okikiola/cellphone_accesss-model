# predicting if they will be able to have cellphone access

import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.model_selection import train_test_split, GridSearchCV 

# preprocessing data
from sklearn.preprocessing import StandardScaler, OneHotEncoder 
from sklearn.impute import SimpleImputer 

# pipelines
from sklearn.pipeline import Pipeline 
from sklearn.compose import ColumnTransformer 

# models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC

# evaluating regression model accuracy
from sklearn.metrics import classification_report, accuracy_score


# loading the dataset
Data = pd.read_csv("C:\\Datasets\\financial inclusion in Africa\\Train.csv")
# dropped the column "uniqueid"
Data.drop(columns=['uniqueid'], inplace=True)
# dropping duplicate values
Data.duplicated().sum()
Data.drop_duplicates(keep='first')
# print first 5
Data.head()

print(Data.columns, '\n')
print(Data.shape)

Data.info()
# Data.describe()

Data.columns

# checking for null values
Data[Data.isnull()].sum()

print(F"Countries in the dataset are:{Data['country'].value_counts()}")
sns.catplot(x="country", kind="count", data=Data)

print(F"bank_account in the dataset are:{Data['bank_account'].value_counts()}")

# Explore Target distribution 
sns.catplot(x="bank_account", kind="count", data=Data)

print(F"years in the dataset are:{Data['year'].value_counts()}")
sns.catplot(x="year", kind="count", data=Data)

print(F"education_level in the dataset are:{Data['education_level'].value_counts()}")
sns.catplot(x="education_level", kind="count", data=Data)

print(F"job_type in the dataset are:{Data['job_type'].value_counts()}")
sns.catplot(x='job_type', kind='count', data=Data)

print(F"marital_status in the dataset are:{Data['marital_status'].value_counts()}")
sns.catplot(x='marital_status', kind='count', data=Data)

print(F"gender_of_respondent in the dataset are:{Data['gender_of_respondent'].value_counts()}")
sns.catplot(x='gender_of_respondent', kind='count', data=Data)

print(F"cellphone_access in the dataset are:{Data['cellphone_access'].value_counts()}")
sns.catplot(x='cellphone_access', kind='count', data=Data)

# features 
X = Data.drop(columns=['cellphone_access'])

# target
y = Data['cellphone_access']

# X.shape
# y.shape

# spliting into train and test set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=42)
# Xtrain.shape

# preprocessing the feature matrics and building pipeline

# selecting the data types
num_features = Xtrain.select_dtypes(include=['float64, int64']).columns
cat_features = Xtrain.select_dtypes(include=['object']).columns

# instantisting the preprocessing model - if i want to use just one model
# num_transformer = StandardScaler()
# cat_transformer = OneHotEncoder(handle_unknown='ignore')

# instantisting the preprocessing model - if i want to use just 2 or more model, do it like this
num_transformer  = Pipeline(steps=[('scaler', StandardScaler()), ('imputer', SimpleImputer(strategy='mean'))])
cat_transformer  = Pipeline(steps=[('encoder', OneHotEncoder(handle_unknown='ignore')), ('imputer', SimpleImputer(strategy='most_frequent'))])

# ColumnTransformer - to preprocess the num/cat_features with the num/cat_transformer
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features), ('cat', cat_transformer, cat_features)])

# pipeline - creating a work flow 
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier(random_state=42))])
pipeline
# training the model
pipeline.fit(Xtrain, ytrain)



# prediction
y_pred = pipeline.predict(Xtest)
print('prediction:', y_pred)
print('actual:', ytest)

score = pipeline.score(Xtrain, ytrain)
print('score:', score)

report = classification_report(ytest, y_pred)
print(report)


# creating a pickle file
import pickle
with open('model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print('model saved as model.pkl')
