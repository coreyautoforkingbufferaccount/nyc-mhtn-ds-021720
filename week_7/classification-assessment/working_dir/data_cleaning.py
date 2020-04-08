import pandas as pd
import numpy as np
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
import pickle
import inspect


def pickle_read(path):
    with open(path, "rb") as f:
        pickle_file = pickle.load(f)
    return pickle_file

def pickle_write(item, path):
    with open(path, "wb") as f:
        pickle.dump(item, f)

def increasing_debt(row, column, i):
    if i > 1 and row[column + f'{i}'] < row[column + f'{i - 1}'] and row["is_streak"] == 1:
        row["debt_streak"] += 1
        row["raw_debt_accum"] += row[column + f'{i - 1}'] - row[column + f'{i}']
    else:
        row["is_streak"] = 0
    return row


def initiate_placeholders(df):
    df["is_streak"], df["debt_streak"] = 1, 0
    df["raw_debt_accum"] = 0
    return df


def remove_placeholders(df):
    return df.drop(columns=["is_streak", "raw_debt_accum"])


def replace_unknowns(df):
    education_dict = {4: 0, 5: 0, 6: 0}
    marriage_dict = {3: 0}
    df["EDUCATION"].replace(education_dict, inplace=True)
    df["MARRIAGE"].replace(marriage_dict, inplace=True)
    return df


def exclude_columns(looped_cols):
    """Gathers column names to exclude"""
    looped_exc = []
    for col in looped_cols:
        sing_exc = [col + f"{i}" for i in np.arange(1, 7)]
        looped_exc.extend(sing_exc)
    looped_exc.extend(["ID", "default payment next month"])
    return looped_exc


def calculate_utilization(df):
    df["avg_utilization"], df["avg_payment_impact"] = 0, 0
    initiate_placeholders(df)
    for i in np.arange(1, 7):
        df['payment_impact' + f'{i}'] = (df['PAY_AMT' + f'{i}']) / df["LIMIT_BAL"]
        df["utilization" + f'{i}'] = df["BILL_AMT" + f'{i}'] / df["LIMIT_BAL"]
        if i > 1:
            df = df.apply(lambda x: increasing_debt(x, "utilization", i), axis=1)
        df["avg_utilization"] += df["utilization" + f'{i}']
        df["avg_payment_impact"] += df["payment_impact" + f'{i}']
    df["avg_utilization"] = df["avg_utilization"] / 6
    df["avg_payment_impact"] = df["avg_payment_impact"] / 6
    df["debt_avg_delta"] = (df["raw_debt_accum"] / df["debt_streak"]).fillna(0)
    df = remove_placeholders(df)
    return df
