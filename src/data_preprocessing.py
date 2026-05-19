import numpy as np
import pandas as pd
from data_loader import load_data_raw, load_data_processed

def combine_data(df1_name: str, df2_name: str):
    df_1 = load_data_raw(df1_name)
    df_2 = load_data_raw(df2_name)
    return pd.concat([df_1, df_2], ignore_index=True)

def clean_data(df):
    df.dropna(inplace=True)
    return df

def train_test_split(df, test_size=0.2, random_state=42, target_col=None):
    np.random.seed(random_state)
    shuffled_indices = np.random.permutation(len(df))
    test_set_size = len(df) * test_size
    test_indices = shuffled_indices[:int(test_set_size)]
    train_indices = shuffled_indices[int(test_set_size):]

    x_train = df.iloc[train_indices].drop(target_col, axis=1).values
    y_train = df.iloc[train_indices][target_col].values
    x_test = df.iloc[test_indices].drop(target_col, axis=1).values
    y_test = df.iloc[test_indices][target_col].values
    return x_train, y_train, x_test, y_test

def normalize_data(x_train, x_test):
    mean_train = np.mean(x_train, axis=0)
    std_train = np.std(x_train, axis=0)
    x_train_norm = (x_train - mean_train) / std_train
    x_test_norm = (x_test - mean_train) / std_train
    return x_train_norm, x_test_norm


