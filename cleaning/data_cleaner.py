import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import field_mapping as fm

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

# replace blank cells and cells with "?" with NaN
def blank_cells(df):
    df.replace("?", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)

# convert strings to lowercase
def lowercase(df):
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].str.lower()
    return df

# one hot encoding 
# reference code: https://www.geeksforgeeks.org/ml-one-hot-encoding/
def one_hot_encoding(data, columns):
    enc = OneHotEncoder(sparse_output=False)
    one_hot_encoded = enc.fit_transform(data[columns])
    one_hot_df = pd.DataFrame(one_hot_encoded, columns=enc.get_feature_names_out(columns))
    df_encoded = pd.concat([data, one_hot_df], axis=1)
    df_encoded = df_encoded.drop(columns, axis=1)
    return df_encoded

# target encoding for variables with large set of categorical data
def target_encoding(df, col_name, outcome, new_col_name):
    field_means = df.groupby(col_name)[outcome].mean()
    df[new_col_name] = df[col_name].map(field_means)
    df.drop(columns=[col_name], inplace=True)

def primary_processor(df):
    df.drop(['wave'], axis = 1, inplace= True)
    df = df.map(strip_byte)
    blank_cells(df)
    lowercase(df)

    # intial categorical variable processing
    target_cols = find_non_numerical(df)
    target_cols.remove('field')
    df_encoded = one_hot_encoding(df, target_cols)
    return df_encoded

def field_processing(df):
    df1_te = df.copy()
    df2_group = df.copy()

    # target encoding for categorical variables with large set of data
    target_encoding(df1_te, 'field', 'match', 'field_target')

    # encoding field with groups
    df2_group['field'] = df2_group['field'].map(fm.field_mapping).fillna('other')
    df2_group = one_hot_encoding(df2_group, ['field'])

    return df1_te, df2_group

# data imputation method
def impute_data(df):
    df2 = df.copy()
    nan_cols = []
    for col in df2.columns:
        if df2[col].isna().any():
            nan_cols.append(col)
    for col in nan_cols:
        median = df2[col].median()
        df2[col] = df2[col].fillna(median)
    return df2

# main function
def main():
    excel_to_csv("../data/original/speed_dating.xlsx", "../data/original/backup.csv")
    df = pd.read_csv("../data/original/backup.csv")

    df_encoded = primary_processor(df)
    df1_te, df2_group = field_processing(df_encoded)

    df1_te_imputed = impute_data(df1_te)
    df2_group_imputed = impute_data(df2_group)

    df1_te.to_csv("../data/cleaned/speeddating_target_encoded_NaN.csv", index=False)
    df2_group.to_csv("../data/cleaned/speeddating_grouped_NaN.csv", index=False)
    
    df1_te_imputed.to_csv("../data/cleaned/speeddating_target_encoded_imputed.csv", index=False)
    df2_group_imputed.to_csv("../data/cleaned/speeddating_grouped_imputed.csv", index=False)

if __name__=="__main__":
    main()