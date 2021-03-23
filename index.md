# Data 310 Midterm Project
### My imports:
```python,  echo = True
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

### My Answer: 96453

```python,  echo = False
data = pd.read_csv('/content/drive/MyDrive/WM/Junior/DATA 310/weatherHistory.csv')
print(np.shape(data))
```

## 2)	In the weatherHistory.csv data how many features are just nominal variables?

### My Answer: 3. Of the variables listed below in the data set, only three are nominal. The rest are continuous.
```python,  echo = True
data.head()
```
