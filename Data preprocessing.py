import kagglehub
import os
import pandas as pd
import numpy as np

### IMPORTING DATASET

path = kagglehub.dataset_download( "shohinurpervezshohan/freelancer-earnings-and-job-trends" )
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

df = pd.read_csv(os.path.join(path, csv_files[0]))

# print(df.columns.tolist())

### DIVING DATASET INTO DEPENDENT AND INDEPENDENT VARIABLE

X = df.drop(["Freelancer_ID", "Earnings_USD"], axis=1)
# removing column
y = df["Earnings_USD"]

# print(X.columns.tolist())

### ENCODING INDEPENDENT VARIABLE

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 10])], remainder = 'passthrough')
X = np.array( ct.fit_transform(X))

# feature_names = ct.get_feature_names_out()
# for i, name in enumerate(feature_names):
#     print(i, name)

# print(X)

### SPLITTING DATASET INTO TRAINING SET AND TEST SET

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 1 )

# print(y_test)

### FEATURE SCALING

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 29:] = sc.fit_transform( X_train[:, 29:] )
X_test[:, 29:] = sc.transform( X_test[:, 29:] )

# print( X_train)