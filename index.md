# Data 310 Midterm Project
### My imports:
```python
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
import matplotlib.pyplot as plt
from sklearn import linear_model as lm
from sklearn.datasets import load_boston
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor
```


## 1)	Import the weatherHistory.csv into a data frame. How many observations do we have?

### My Answer
96453

### Correct Answer
96453

```python
data = pd.read_csv('/content/drive/MyDrive/WM/Junior/DATA 310/weatherHistory.csv')
print(np.shape(data))
```

## 2)	In the weatherHistory.csv data how many features are just nominal variables?

### My Answer
3; Of the variables listed below in the data set (Formatted Date, Summary, Precip Type,	Temperature, Apparent Temperature, Humidity, Wind Speed, Wind Bearing, Visibility, Loud Cover	Pressure, Daily Summary), only three are nominal (Summary, Precip Type, and Daily Summary). The rest are continuous.

### Correct Answer
3


## 3) If we want to use all the unstandardized observations for 'Temperature (C)' and predict the Humidity, the resulting root mean squared error is (just copy the first 4 decimal places)

### My Answer
0.1514

### Correct Answer: 
0.1514

### My Code:
```python
model = LinearRegression()
x = data[['Temperature (C)']]
y = data['Humidity']
model.fit(x, y)
yprime = model.predict(data['Temperature (C)'].values.reshape(-1, 1))
residuals = [0]*len(yprime)
for i in range(len(yprime)):
  residuals[i] = (yprime[i] - y[i])**2
np.sqrt(np.mean(residuals))          #Final RMSE Output.
```
This resulted in an RMSE of about 0.15144.

## 4) If the input feature is the Temperature and the target is the Humidity and we consider 20-fold cross validations with random_state = 2020, the Ridge model with alpha = 0.1 and standardize the input train and the input test data. The average RMSE on the test sets is (provide your answer with the first 6 decimal places):

### My Answer
0.151438

### Correct Answer
0.151438

### My Code:
```python
kf = KFold(n_splits=20, random_state=2020, shuffle=True)
MSE_train = []
MSE_test = []
ss = StandardScaler()
xs = ss.fit_transform(data[['Temperature (C)']])
y = data['Humidity']
for idxtrain, idxtest in kf.split(xs):
  model = Ridge(alpha = 0.1)
  x_train = xs[idxtrain]
  x_test = xs[idxtest]
  y_train = y[idxtrain]
  y_test  = y[idxtest]
  model.fit(x_train,y_train)
  yhat_train = model.predict(x_train)
  yhat_test = model.predict(x_test)
  MSE_train.append(MSE(y_train,yhat_train))
  MSE_test.append(MSE(y_test,yhat_test))
test_rmse = np.sqrt(MSE_test)
print(MSE_test)                    #For checking purposes, all MSEs
print(test_rmse)                   #For checking purposes, all RMSEs
print(np.mean(test_rmse))          #Final Mean RMSE Output.
```
This resulted in a mean RMSE of about 0.151438.

## 5) Suppose we want to use Random Forrest with 100 trees and max_depth=50 to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-cross validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 6 decimal places):

### My Answer
0.143508

### Correct Answer
0.143508

### My Code:
```python
kf = KFold(n_splits=10, random_state=1693, shuffle=True)
MSE_train = []
MSE_test = []
x = data['Apparent Temperature (C)'].values.reshape(-1,1)
y = data['Humidity']
for idxtrain, idxtest in kf.split(x):
  model = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=1693)
  x_train = x[idxtrain]
  x_test = x[idxtest]
  y_train = y[idxtrain]
  y_test  = y[idxtest]
  model.fit(x_train,y_train)
  yhat_train = model.predict(x_train)
  yhat_test = model.predict(x_test)
  MSE_train.append(MSE(y_train,yhat_train))
  MSE_test.append(MSE(y_test,yhat_test))
test_rmse = np.sqrt(MSE_test)
print(MSE_test)                    #For checking purposes, all MSEs
print(test_rmse)                   #For checking purposes, all RMSEs
print(np.mean(test_rmse))          #Final Mean RMSE Output.
```
This resulted in a mean RMSE of about 0.143508. 

## 6) Suppose we want use polynomial features of degree 6 and we want to predict the Humidity with the Apparent Temperature and we want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1693) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 5 decimal places):

### My Answer
0.378765

### Correct Answer
0.14346

### My Code:
```python
MSE_train = []
MSE_test = []
x = data['Apparent Temperature (C)'].values.reshape(-1,1)
y = data['Humidity']
kf = KFold(n_splits=10, random_state=1693,shuffle=True)
for idxtrain, idxtest in kf.split(x):
  model = LinearRegression()
  x_train = x[idxtrain]
  x_test = x[idxtest]
  y_train = y[idxtrain]
  y_test  = y[idxtest]
  polynomial_features= PolynomialFeatures(degree=6)
  x_poly_train = polynomial_features.fit_transform(np.array(x_train))
  x_poly_test = polynomial_features.fit_transform(np.array(x_test))
  model.fit(x_poly_train,y_train)
  yhat_train = model.predict(x_poly_train)
  yhat_test = model.predict(x_poly_test)
  MSE_train.append(np.sqrt(MSE(y_train,yhat_train)))
  MSE_test.append(np.sqrt(MSE(y_test,yhat_test)))
test_rmse = np.sqrt(MSE_test)
print(MSE_test)                    #For checking purposes, all MSEs
print(test_rmse)                   #For checking purposes, all RMSEs
print(np.mean(MSE_test))           #For checking purposes, Final Mean MSE Output.
print(np.mean(test_rmse))          #Final Mean RMSE Output.
```
This resulted in a mean RMSE of about 0.378765. However, the correct answer seems to have been the resultant mean MSE: 0.14346.

## 7)  If the input feature is the Temperature and the target is the Humidity and we consider 10-fold cross validations with random_state=1234, the Ridge model with alpha =0.2. Inside the cross-validation loop standardize the input data. The average RMSE on the test sets is (provide your answer with the first 4 decimal places):

### My Answer
0.1514

### Correct Answer
0.151444

### My Code:
```python
kf = KFold(n_splits=10, random_state=1234, shuffle=True)
MSE_train = []
MSE_test = []
y = data['Humidity']
x = data[['Temperature (C)']]
for idxtrain, idxtest in kf.split(x):
  model = Ridge(alpha = 0.2)
  ss = StandardScaler()
  xs = ss.fit_transform(x)
  x_train = xs[idxtrain]
  x_test = xs[idxtest]
  y_train = y[idxtrain]
  y_test  = y[idxtest]
  model.fit(x_train,y_train)
  yhat_train = model.predict(x_train)
  yhat_test = model.predict(x_test)
  MSE_train.append(MSE(y_train,yhat_train))
  MSE_test.append(MSE(y_test,yhat_test))
test_rmse = np.sqrt(MSE_test)
print(MSE_test)                    #For checking purposes, all MSEs
print(test_rmse)                   #For checking purposes, all RMSEs
print(np.mean(test_rmse))          #Final Mean RMSE Output.
```
This resulted in a mean RMSE of about 0.151444.

## 8)  Suppose we use polynomial features of degree 6 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is (provide your answer with the first 4 decimal places):

### My Answer
2.4736

### Correct Answer
6.0234

### My Code:
```python
MSE_train = []
MSE_test = []
y = data['Temperature (C)']
x = data[['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']]
x = x.values
kf = KFold(n_splits=10, random_state=1234,shuffle=True)
for idxtrain, idxtest in kf.split(x):
  model = LinearRegression()
  x_train = x[idxtrain]
  x_test = x[idxtest]
  y_train = y[idxtrain]
  y_test  = y[idxtest]
  polynomial_features= PolynomialFeatures(degree=6)
  x_poly_train = polynomial_features.fit_transform(np.array(x_train))
  x_poly_test = polynomial_features.fit_transform(np.array(x_test))
  model.fit(x_poly_train,y_train)
  yhat_train = model.predict(x_poly_train)
  yhat_test = model.predict(x_poly_test)
  MSE_train.append(np.sqrt(MSE(y_train,yhat_train)))
  MSE_test.append(np.sqrt(MSE(y_test,yhat_test)))
test_rmse = np.sqrt(MSE_test)
print(MSE_test)                    #For checking purposes, all MSEs
print(test_rmse)                   #For checking purposes, all RMSEs
print(np.mean(MSE_test))           #For checking purposes, Final Mean MSE Output.
print(np.mean(test_rmse))          #Final Mean RMSE Output.
```
This resulted in a mean RMSE of about 2.47368. Again, the correct answer seems to be closer to my caclulated mean MSE output, 6.11920, though not exactly the same.

## 9)  Suppose we use Random Forest with 100 trees and max_depth=50 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 4 decimal places):

### My Answer
5.8297

### Correct Answer
5.8323

### My Code:
```python
kf = KFold(n_splits=10, random_state=1234, shuffle=True)
MSE_train = []
MSE_test = []
y = data['Temperature (C)']
x = data[['Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)']]
x = x.values
for idxtrain, idxtest in kf.split(x):
  model = RandomForestRegressor(n_estimators=100, max_depth=50, random_state=1234) #I set the random state here, which might have thrown off the final result.
  x_train = x[idxtrain]
  x_test = x[idxtest]
  y_train = y[idxtrain]
  y_test  = y[idxtest]
  model.fit(x_train,y_train)
  yhat_train = model.predict(x_train)
  yhat_test = model.predict(x_test)
  MSE_train.append(MSE(y_train,yhat_train))
  MSE_test.append(MSE(y_test,yhat_test))
test_rmse = np.sqrt(MSE_test)
print(MSE_test)                    #For checking purposes, all MSEs
print(test_rmse)                   #For checking purposes, all RMSEs
print(np.mean(test_rmse))          #Final Mean RMSE Output.
```
This resulted in a mean RMSE of about 5.8297.

## 9)  Suppose we use Random Forest with 100 trees and max_depth=50 and we want to predict the Temperature by using 'Humidity','Wind Speed (km/h)','Pressure (millibars)','Wind Bearing (degrees)' We want to estimate the root mean squared error by using 10-fold cross-validations (random_state=1234) and computing the average of RMSE on the test sets. The result we get is  (provide your answer with the first 4 decimal places):

### My Answer
Decreasing.

### Correct Answer
Decreasing.

### My Code:
```python
plt.scatter(data['Humidity'], data['Temperature (C)'])
```
### Resultant Graph:
![HumidityvTemp](https://jocain.github.io/Data-310-Midterm/midtermgraph.png)
