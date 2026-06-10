import numpy as np

### PREPROCESSING DATASET

from preprocess import data_preprocessor
X_train, X_test, y_train, y_test = data_preprocessor() 

### FEATURE SCALING

from sklearn.preprocessing import StandardScaler
sc_y = StandardScaler()
y_train = sc_y.fit_transform( y_train.to_numpy().reshape(-1, 1))

### TRAINING SVR MODEL

from sklearn.svm import SVR
regressor = SVR( kernel = 'rbf')
regressor.fit(X_train, y_train.ravel())

###  PREDICTING TEST SET

y_pred = sc_y.inverse_transform( regressor.predict(X_test).reshape(-1,1))
np.set_printoptions( precision = 2 )

# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.to_numpy().reshape(len(y_test),1)),1))

### CHECKING MODEL PERFORMANCE

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)