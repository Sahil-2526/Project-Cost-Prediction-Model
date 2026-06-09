import kagglehub
import os
import pandas as pd
import numpy as np

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

# print(X)







