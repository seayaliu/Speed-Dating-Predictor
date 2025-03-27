import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# strip "b''" byte wrappings from data set items
def strip_byte(item):
    if isinstance(item, str) and item.startswith("b'") and item.endswith("'"):
        return item[2:-1]
    return item

# convert excel to csv
def excel_to_csv(file_path, file_name):
    df = pd.read_excel(file_path, index_col=None)
    df.to_csv(file_name, index=False)  

# load data set from file
def load_data(file_path):
    df = pd.read_csv(file_path)
    df = df.map(strip_byte)
    return df 

# main function
csv = excel_to_csv("speeddating.xlsx", "backup.csv")
data = load_data('backup.csv')

if not os.path.exists("speeddating.csv"):
    data.to_csv("speeddating.csv", index=False)
    print("New 'speeddating.csv' created and saved.")
else:
    print("already exists")

df = pd.DataFrame(data)
# print(df.head())

