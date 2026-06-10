import numpy as np

### PREPROCESSING DATASET

from preprocess import data_preprocessor
X_train, X_test, y_train, y_test = data_preprocessor() 


### TRAINING DECISION TREE REGRESSOR MODEL

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)
regressor.fit(X_train, y_train)

###  PREDICTING TEST SET

y_pred = regressor.predict(X_test)
np.set_printoptions( precision = 2 )

# print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.to_numpy().reshape(len(y_test),1)),1))

### CHECKING MODEL PERFORMANCE

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(r2)