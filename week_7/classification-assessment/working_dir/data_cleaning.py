import pandas as pd
import numpy as np
import pickle


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

def exclude_columns(looped_cols, start, end):
    """Gathers column names to exclude"""
    looped_exc = []
    for col in looped_cols:
        sing_exc = [col + f"{i}" for i in np.arange(start, end)]
        looped_exc.extend(sing_exc)
    return looped_exc


def extract_dummies(df, column, value):
    """Creates a column with dummy variables for matches in a value"""

    if np.isnan(value):
        return np.where(df[column].isna().values == True, 1, 0)
    else:
        return np.where(df[column].values == value, 1, 0)


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


def split_pay_columns(df):
    """Extracts the quantitative information (The number of months of missed payments)
    from the qualitative, the two fields that determined ontime payments"""

    df = df.copy()
    for i in np.arange(0, 1):
        column = "PAY_" + f"{i}"
        df[column] = df[column].astype(int)
        dflt = df[column].unique().tolist()
        default_vals = dict(zip(dflt, dflt))
        default_vals[-1], default_vals[-2] = 0, 0
        df[f"{column}N1"] = extract_dummies(df, column, -1)
        df[f"{column}N2"] = extract_dummies(df, column, -1)
        df[column] = df[column].map(default_vals)
    return df

def combine_pay_columns(df):
    """Extracts non correlated information from the past history pay columns. Any
    time there is an improvement to this field it is incremented and the sum is added
    to a new column in the dataframe."""

    df = df.copy()
    before= df["PAY_0"].values
    results = np.zeros(before.size)
    for i in np.arange(2, 7):
        column = "PAY_" + f"{i}"
        after = df[column].values
        comparison = before < after
        results += comparison.astype(int)
    df["payment_improvements"] = results
    return df