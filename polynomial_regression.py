import numpy as np

### PREPROCESSING DATASET

from preprocess import data_preprocessor
X_train, X_test, y_train, y_test = data_preprocessor() 

### TRAINING PLOYNOMIAL REGRESSION MODEL

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit( X_poly, y_train)

###  PREDICTING TEST SET

y_pred = regressor.predict(poly_reg.transform(X_test))
np.set_printoptions( precision = 2 )

# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.to_numpy().reshape(len(y_test),1)),1))

### CHECKING MODEL PERFORMANCE

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)