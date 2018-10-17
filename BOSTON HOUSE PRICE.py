#import neccesary library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import dataset
dataset = pd.read_csv('housing.csv')
x = dataset.iloc[:,:3].values
y = dataset.iloc[:,3:4].values

#spplit dataset
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(x,y,random_state =0) 

#Apply feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
#feature scale the x data
xtrain = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)
#feature scale the y data
ytrain = sc.fit_transform(ytrain)
ytest = sc.transform(ytest)

#APPLY THE POLYNOMIAL REGRESSION MODEL 
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly = PolynomialFeatures(degree = 3)
x_train_poly = poly.fit_transform(xtrain) #CHANGE DATASET IN  POLYNOMIAL TERM
x_test_poly = poly.transform(xtest)

regressor = LinearRegression() #USE LINEAR REGRESSION MODEL FOR THE FIT AND PREDICT OUTPUT
regressor.fit(x_train_poly,ytrain)

#PREDICT THE OUTPUT FOR  X TEST DATASET
y_pred = regressor.predict(x_test_poly)


#CHECK THE ACCURACY BY THE R2 METHOD AND IT HAVE 81% ACCURACY
#SSRES = SUM((YTEST - YPRDICT)^2)
ssres = np.sum(np.square(ytest - y_pred))
#SSTOT = SUM((YTEST - YAVG)^2)
sstot = np.sum(np.square(ytest - np.mean(ytest)))

#EQUATIN FOR R2 = 1- (SSRES/SSTOT)
r = 1-(ssres/sstot)

