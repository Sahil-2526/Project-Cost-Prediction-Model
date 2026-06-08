import kagglehub
import os
import pandas as pd

path = kagglehub.dataset_download( "shohinurpervezshohan/freelancer-earnings-and-job-trends" )
# downloads the Kaggle dataset and returns the folder location

csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
# keeps only files ending with .csv
# os.listdir(path) :- returns all files in the dataset folder

print("CSV files found:", csv_files)

df = pd.read_csv(os.path.join(path, csv_files[0]))
# path :- path to the file
# csv_files[0] :- name of dataset file 
# os.path.join( ) :- used join path

print(df.shape)
# gives rows and columns
print(df.head())
# gives first 5 rows 