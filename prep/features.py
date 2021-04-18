import pandas as pd


def cabin_prefix(train, test):
    train['Cabin'].fillna('9999', inplace=True)
    test['Cabin'].fillna('9999', inplace=True)
    train['CabinPrefix'] = train['Cabin'].apply(lambda x : x[0])
    test['CabinPrefix'] = test['Cabin'].apply(lambda x : x[0])
    return train, test


def ticket_prefix(train, test):
    train['Ticket'] = train['Ticket'].map(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'X')
    test['Ticket'] = test['Ticket'].map(lambda x: str(x).split()[0] if len(str(x).split()) > 1 else 'X')
    return train, test


def name_process(train, test):
    full = pd.concat([train, test]).reset_index(drop=True)
    full['FirstName'] = full.Name.map(lambda x: str(x).split(',')[0])
    full['Surname'] = full.Name.map(lambda x: str(x).split(',')[1])
    for col in ['Name', 'FirstName', 'Surname']:
        full['Counter_' + col] = full[col].map(full.groupby(col)['PassengerId'].count().to_dict())
    full.drop(columns = ['Name', 'Surname'], inplace = True)
    return full[:train.shape[0]], full[train.shape[0]:]
