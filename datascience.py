# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 21:07:30 2018

@author: Dell
"""
from math import sqrt
from scipy.optimize import fmin_l_bfgs_b
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import accuracy_score, r2_score,confusion_matrix, classification_report, precision_score,recall_score, f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

dataset=pd.read_csv('E:/Fast Semesters/6th Semester Books and Slides/DATA SCIENCE/Project1/final.csv',encoding="ISO-8859-1")


# Data Preprocessings

dataset['Coconut Oil Price (US Dollars per Metric Ton)']=dataset['Coconut Oil Price (US Dollars per Metric Ton)'].fillna(dataset['Coconut Oil Price (US Dollars per Metric Ton)'].mean())
dataset['Olive Oil, extra virgin Price (US Dollars per Metric Ton)']=dataset['Olive Oil, extra virgin Price (US Dollars per Metric Ton)'].fillna(dataset['Olive Oil, extra virgin Price (US Dollars per Metric Ton)'].mean())
dataset['Sunflower oil Price (US Dollars per Metric Ton)']=dataset['Sunflower oil Price (US Dollars per Metric Ton)'].fillna(dataset['Sunflower oil Price (US Dollars per Metric Ton)'].mean())
dataset['Rapeseed Oil Price (US Dollars per Metric Ton)']=dataset['Rapeseed Oil Price (US Dollars per Metric Ton)'].fillna(dataset['Rapeseed Oil Price (US Dollars per Metric Ton)'].mean())
dataset['Rapeseed Oil Price (US Dollars per Metric Ton)']=dataset['Rapeseed Oil Price (US Dollars per Metric Ton)'].fillna(dataset['Rapeseed Oil Price (US Dollars per Metric Ton)'].mean())
dataset['Exchange Rate']=dataset['Exchange Rate'].fillna(dataset['Exchange Rate'].mean())

PalmOil= dataset['Palm oil Price (US Dollars per Metric Ton)']
Soybean= dataset['Soybean Oil Price (US Dollars per Metric Ton)']
Coconut= dataset['Coconut Oil Price (US Dollars per Metric Ton)']
Olive= dataset['Olive Oil, extra virgin Price (US Dollars per Metric Ton)']
Sunflower= dataset['Sunflower oil Price (US Dollars per Metric Ton)']
Rapeseed=dataset['Rapeseed Oil Price (US Dollars per Metric Ton)']
CrudeOil=dataset['Crude Oil (petroleum) Price (US Dollars per Barrel)']
ExchangeRate=dataset['Exchange Rate']

#""Pearson Correlation Coefficient
print("Pearson Correlation Coefficient \n")
SoybeanCoR=np.corrcoef(PalmOil, Soybean)[0, 1]
CoconutCoR=np.corrcoef(PalmOil, Coconut)[0, 1]
OliveCoR=np.corrcoef(PalmOil, Olive)[0, 1]
SunflowerCoR=np.corrcoef(PalmOil, Sunflower)[0, 1]
RapeseedCoR=np.corrcoef(PalmOil, Rapeseed)[0, 1]
CrudeOilCoR=np.corrcoef(PalmOil, CrudeOil)[0, 1]
ExchangeRateCoR=np.corrcoef(PalmOil, ExchangeRate)[0, 1]
print("Soybean ",SoybeanCoR)
print("Coconut ",CoconutCoR)
print("Olive ",OliveCoR)
print("Sunflower ",SunflowerCoR)
print("Rapeseed ",RapeseedCoR)
print("CrudeOil ",CrudeOilCoR)
print("ExchangeRate ",ExchangeRateCoR)

#ALgos

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

array = dataset.values
X = array[:, 1:7]  
Y = array[:,8]  
Y=Y.astype('int')
t_size = 0.30  # test size 30%
seed = 7

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, random_state=seed, test_size=t_size)  # Cross Validation

#SVR
print("\nSequential Minimal Optimization")
svm=SVR(kernel='linear').fit(X_train,Y_train)
predictions = svm.predict(X_test)
#print(predictions)
print("Prediction Length : ",len(predictions))
acc1=r2_score(Y_test, predictions)
MSE= mean_squared_error(predictions, np.array(list(Y_test)))
MAE= mean_absolute_error(Y_test, predictions)
MAPE= mean_absolute_percentage_error(Y_test, predictions)
print("MAE ",MAE)
print("MSE ",MSE)
print("MAPE ",MAPE)
print("Accuracy : ",acc1)

#MLP
print("\nMulti-layer perceptron regressor")
mlp=MLPRegressor().fit(X_train,Y_train)
predictionsmlp = mlp.predict(X_test)
#print(predictions)
print("Prediction Length : ",len(predictionsmlp))
acc=r2_score(Y_test, predictionsmlp)
MSE= mean_squared_error(predictionsmlp, np.array(list(Y_test)))
MAE= mean_absolute_error(Y_test, predictionsmlp)
MAPE= mean_absolute_percentage_error(Y_test, predictionsmlp)

print("MAE : ",MAE)
print("MSE : ",MSE)
print("MAPE : ",MAPE)
print("Accuracy : ",acc)



print("\nSome Extra Algorithms  :")

#KNN
print("\nKNeighbors ")
knn = KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictionsknn = knn.predict(X_test)
acc=r2_score(Y_test, predictionsknn)
MSE= mean_squared_error(predictionsknn, np.array(list(Y_test)))
MAE= mean_absolute_error(Y_test, predictionsknn)
MAPE= mean_absolute_percentage_error(Y_test, predictionsknn)
print("MAE : ",MAE)
print("MSE : ",MSE)
print("MAPE : ",MAPE)
print("Accuracy : ",acc)

#Random Forest
print("\nRandom Forest ")
rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
predictionsrf = rf.predict(X_test)
acc=r2_score(Y_test, predictionsrf)
MSE= mean_squared_error(predictionsrf, np.array(list(Y_test)))
MAE= mean_absolute_error(Y_test, predictionsrf)
MAPE= mean_absolute_percentage_error(Y_test, predictionsrf)
print("MAE : ",MAE)
print("MSE : ",MSE)
print("MAPE : ",MAPE)
print("Accuracy : ",acc)

#Random Forest
print("\nNaive Bayes")
nb = GaussianNB()
nb.fit(X_train,Y_train)
predictionsnb = nb.predict(X_test)
acc=r2_score(Y_test, predictionsnb)
MSE= mean_squared_error(predictionsnb, np.array(list(Y_test)))
MAE= mean_absolute_error(Y_test, predictionsnb)
MAPE= mean_absolute_percentage_error(Y_test, predictionsnb)
print("MAE : ",MAE)
print("MSE : ",MSE)
print("MAPE : ",MAPE)
print("Accuracy : ",acc)

print("After research the algorithm which gives best result is Sequential Minimal Optimization(SVR) with accuracy", acc1)