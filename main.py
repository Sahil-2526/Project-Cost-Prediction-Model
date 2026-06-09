import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download( "shohinurpervezshohan/freelancer-earnings-and-job-trends" )
csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]

print("CSV files found:", csv_files)

df = pd.read_csv(os.path.join(path, csv_files[0]))

# Dividing dataset into dependent and independent variables

X = df.drop(["Freelancer_ID", "Earning_USD"], axis=1)
# removing column
y = df["Earning_USD"]

