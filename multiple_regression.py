### PREPROCESSING DATASET

from preprocess import data_preprocessor
X_train, X_test, y_train, y_test = data_preprocessor() 

### TRAINING MULTIPLE REGRESSION MODEL

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.train( X_train, y_train)