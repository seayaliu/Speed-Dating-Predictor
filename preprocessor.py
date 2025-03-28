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

# find categorical variables in data set
def find_non_numerical(data):
    str_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in str_cols:
        if data[col].str.isnumeric().all():
            data[col] = data[col].astype(int)
    str_cols = data.select_dtypes(include=['object']).columns.tolist()
    return str_cols

def blank_cells(df):
    df.replace("?", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)

# one hot encoding 
# reference code: https://www.geeksforgeeks.org/ml-one-hot-encoding/
def one_hot_encoding(data, columns):
    enc = OneHotEncoder(sparse_output=False)
    one_hot_encoded = enc.fit_transform(data[columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(columns))
    df_encoded = pd.concat([data, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(columns, axis=1)
    return df_encoded

# main function
def main():
    excel_to_csv("speed_dating.xlsx", "backup.csv")
    df = pd.read_csv("backup.csv")
    df = df.map(strip_byte)
    blank_cells(df)
    target_cols = find_non_numerical(df)
    target_cols.remove('field')
    df_encoded = one_hot_encoding(df, target_cols)
    print(df_encoded.columns.tolist())

    if not os.path.exists("speeddating.csv"):
        df_encoded.to_csv("speeddating.csv", index=False)
        print("New 'speeddating.csv' created and saved.")
    else:
        print("already exists")

if __name__=="__main__":
    main()