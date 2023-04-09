"""
Original file is located at
    https://colab.research.google.com/drive/1nRrKhl3nHd3TtqX5cYbkfXOTSVG2oIxZ
"""

import numpy as np
import pandas as pd
import joblib

# For Visualization
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
def warns(*args,**kwargs): pass
warnings.warn=warns

# from google.colab import drive
# drive.mount('/content/drive')

df = pd.read_csv('C:/Users/deeks/projects/helloworld/minorproject/crop_recommendation.csv')
print('Data Shape: ', df.shape)

# print(df.tail(25))

# print(df.isna().sum())

# Unique Name of the Crops in Dataset
crops = df['label'].unique()
print(crops.sort())
print ("Total Number of Crops Data: ", len(crops) )
print("\n","-"*20, " List of Crops ", "-"*20)
print(crops.tolist())

#Columns Name
# print("here")
# print(df.columns)

# Features Selection
selected_features = {'N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall'}


# """**FEATURES VISUALISATION**

# COMPARISION OF EACH FEATURES WITH CROPS
# """

# def crop_relation_visual(yfeature):
#     ax = sns.set_style('whitegrid')
#     plt.subplots(figsize=(15,8))

#     ax = sns.barplot(x="label", y=yfeature, data=df, ci=None)
#     # ax.bar_label(ax.containers[0], fontsize=12)

#     plt.xticks(rotation=90, fontsize=12)
#     plt.yticks(rotation=0, fontsize=12)
#     plt.title("Crops Relation with " + str(yfeature), fontsize = 24)
#     plt.xlabel("Crops Name", fontsize = 18)
#     plt.ylabel("values of " + str(yfeature), fontsize = 18)

# for x in selected_features:
#     crop_relation_visual(x)

# """**Statistic Visualization of each Feature of the Crops** """

# # Boxplot for Statistic Viusalization of each Features
# def crop_boxplot_visual(yfeature):
#     ax = sns.set_style('whitegrid')
#     plt.subplots(figsize=(15,8))
#     sns.boxplot(x=yfeature, y="label", data=df)

#     plt.title("Crops Relation with " + str(yfeature), fontsize = 24)
#     plt.xlabel("values of " + str(yfeature), fontsize = 18)
#     plt.ylabel("Crops Name", fontsize = 18)

# for x in selected_features:
#     crop_boxplot_visual(x)

"""**MODELING**"""

xdf = df.copy()

from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LinearRegression as lr
lblencoder= LabelEncoder() # initializing an object of class LabelEncoder

#Fit and Transforming the label column.
lblencoder.fit(xdf['label'])
xdf['label_codes']=lblencoder.transform(xdf['label'])



y = xdf['label_codes'] # Targeted Values Selection
X = xdf[selected_features] # Independent Values


# Data Splitting
from sklearn.metrics import mean_squared_error, accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


key = dict(zip(lblencoder.classes_,range(len(lblencoder.classes_))))
print(key)


"""**Prediction and Accuracy Measure**"""
# #Linear Regression
# reg = lr()
  
# # train the model using the training sets
# reg.fit(X_train, y_train)
# print(reg.score(X_test, y_test))




#RandomForest Classifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
oversample = SMOTE()

features, labels = oversample.fit_resample(xdf.drop(["label", "label_codes"], axis=1), xdf['label_codes'])


rf = RandomForestClassifier(n_estimators=150, criterion='gini',)

# Fit Dataset in Model
rf.fit(X_train, y_train)

print("rf score:",rf.score(X_test, y_test))


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("knn:",knn.score(X_test, y_test))


# newdata=[[90,42,43,20.87974371,82.00274423,6.502985292000001,202.9355362]]
# ans=knn.predict(newdata)
# print(ans)
# ans=list(round(i) for i in ans)
# print(lblencoder.inverse_transform(ans))

# save the model to disk
filename = 'finalized_model.sav'
joblib.dump(rf, filename)
