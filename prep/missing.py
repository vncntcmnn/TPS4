from numpy import testing
import pandas as pd


def cat_na_string(train, test, col):
    train[col] = train[col].fillna('na')
    test[col] = test[col].fillna('na')
    return train, test

def cat_na_mode(train, test, col):
    train[col] = train[col].fillna((train[col].mode()))
    test[col] = test[col].fillna((train[col].mode()))
    return train, test

def cont_na_med(train, test, col, train_only=True):
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)
    return train, test

def cont_na_med(train, test, col, train_only=True):
    mean = train[col].mean()
    train[col] = train[col].fillna(mean)
    test[col] = test[col].fillna(mean)
    return train, test

def fill_age_pclass_mean(train, test):
    full = pd.concat([train, test]).reset_index(drop=True)
    c_map = full[['Age', 'Pclass']].dropna().groupby('Pclass').mean().to_dict()
    full['Age'] = full['Age'].fillna(full['Pclass'].map(c_map['Age']))
    return full[:train.shape[0]].reset_index(drop=True), full[train.shape[0]:].reset_index(drop=True)

def fill_fare_pclass_mean(train, test):
    full = pd.concat([train, test]).reset_index(drop=True)
    c_map = full[['Fare', 'Pclass']].dropna().groupby('Pclass').mean().to_dict()
    full['Fare'] = full['Fare'].fillna(full['Pclass'].map(c_map['Fare']))
    return full[:train.shape[0]].reset_index(drop=True), full[train.shape[0]:].reset_index(drop=True)