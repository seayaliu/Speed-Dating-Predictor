import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import field_mapping as fm

# convert excel to csv
def excel_to_csv(file_path, file_name):
    df = pd.read_excel(file_path, index_col=None)
    df.to_csv(file_name, index=False)  

# strip "b''" byte wrappings from data set items
def strip_byte(item):
    if isinstance(item, str) and item.startswith("b'") and item.endswith("'"):
        return item[2:-1]
    return item

# find categorical variables in data set
def find_non_numerical(data):
    str_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in str_cols:
        if data[col].str.isnumeric().all():
            data[col] = data[col].astype(int)
    str_cols = data.select_dtypes(include=['object']).columns.tolist()
    return str_cols

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

# replace blank cells and cells with "?" with NaN
def blank_cells(df):
    df.replace("?", np.nan, inplace=True)
    df.replace("", np.nan, inplace=True)

# modular primary processing component
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

# processing the categorical variable field
def field_processing(df):
    df1_te = df.copy()
    df2_group = df.copy()

    # target encoding for categorical variables with large set of data
    target_encoding(df1_te, 'field', 'match', 'field_target')

    # encoding field with groups
    df2_group['field'] = df2_group['field'].map(fm.field_mapping).fillna('other')
    df2_group = one_hot_encoding(df2_group, ['field'])

    return df1_te, df2_group

# data median imputation method
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

# standardizing data
def standardize_data(df):
    binoms = []
    for col in df.columns:
        binom = set(df[col].unique()).issubset({0, 1})
        if binom == True:
            binoms.append(col)
    non_binoms = [col for col in df.columns if col not in binoms]
    X_c = df.copy()
    X_c[non_binoms] = (X_c[non_binoms] - X_c[non_binoms].mean(axis=0)) / X_c[non_binoms].std(axis=0, ddof=1)
    return X_c

# undersampling data to balance the dataset
def mod_matches(df, split):
    match1 = df[df["match"] == 1]
    match0 = df[df["match"] == 0]

    n = len(match1) // split
    n0 = n*(10-split)
    sample_match0 = match0.sample(n0, random_state=42)

    df_balanced = pd.concat([match1, sample_match0]).sample(frac=1, random_state=42).reset_index(drop=True)

    return df_balanced

# main function
def main():
    # extract files
    excel_to_csv("../data/original/speed_dating.xlsx", "../data/original/backup.csv")
    df = pd.read_csv("../data/original/backup.csv")

    # preliminary processing
    df_encoded = primary_processor(df)
    df1_te, df2_group = field_processing(df_encoded)    # field processing

    # median impute missing values
    df1_te_imputed = impute_data(df1_te)
    df2_group_imputed = impute_data(df2_group)

    # undersample data
    df2_group_imputed_5050 = mod_matches(df2_group_imputed, 5)
    df2_group_imputed_4060 = mod_matches(df2_group_imputed, 4)
    df2_group_imputed_3070 = mod_matches(df2_group_imputed, 3)

    # scale data
    df2_group_imputed_scaled = standardize_data(df2_group_imputed)

    # save processed data as csvs
    df1_te.to_csv("../data/cleaned/speeddating_target_encoded_NaN.csv", index=False)
    df2_group.to_csv("../data/cleaned/speeddating_grouped_NaN.csv", index=False)
    
    df1_te_imputed.to_csv("../data/cleaned/speeddating_target_encoded_imputed.csv", index=False)
    df2_group_imputed.to_csv("../data/cleaned/speeddating_grouped_imputed.csv", index=False)

    df2_group_imputed_5050.to_csv("../data/cleaned/speeddating_grouped_imputed_balanced5050.csv", index=False)
    df2_group_imputed_4060.to_csv("../data/cleaned/speeddating_grouped_imputed_balanced4060.csv", index=False)
    df2_group_imputed_3070.to_csv("../data/cleaned/speeddating_grouped_imputed_balanced3070.csv", index=False)

    df2_group_imputed_scaled.to_csv("../data/cleaned/speeddating_grouped_imputed_scaled.csv", index=False)

if __name__=="__main__":
    main()