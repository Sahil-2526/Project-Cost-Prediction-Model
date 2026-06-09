import kagglehub
import os
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def data_preprocessing():
    # Importing data
    path = kagglehub.dataset_download( "shohinurpervezshohan/freelancer-earnings-and-job-trends" )
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(path, csv_files[0]))

    # Dividing data into independent and dependent variable
    X = df.drop(["Freelancer_ID", "Earnings_USD"], axis=1)
    y = df["Earnings_USD"]
    
    # Encoding independent data
    ct = ColumnTransformer( transformers = [('encoder', OneHotEncoder(), [0, 1, 2, 3, 4, 10])], remainder = 'passthrough')
    X = np.array( ct.fit_transform(X))
    
    # Dividing dataset into training and test set
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.2, random_state = 1 )
    
    # Feature scaling
    sc = StandardScaler()
    X_train[:, 29:] = sc.fit_transform( X_train[:, 29:] )
    X_test[:, 29:] = sc.transform( X_test[:, 29:] )
    
    return X_train, X_test, y_train, y_test