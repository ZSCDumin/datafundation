#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.simplefilter('ignore')

import json
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import StackingClassifier
from model import get_base_model

train = pd.read_csv('train/train_dataset.csv', sep='\t')
test = pd.read_csv('test/test_dataset.csv', sep='\t')

train['risk_label'].value_counts(dropna=False)

for f in ['user_name', 'action', 'auth_type', 'ip',
          'ip_location_type_keyword', 'ip_risk_level', 'location', 'client_type',
          'browser_source', 'device_model', 'os_type', 'os_version',
          'browser_type', 'browser_version', 'bus_system_code', 'op_target']:
    for v in train[f].unique():
        print(f, v, train[train[f] == v]['risk_label'].mean())
    print('=' * 50)

data = pd.concat([train, test])
print(data.shape)

data['location_first_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['first_lvl'])
data['location_sec_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['sec_lvl'])
data['location_third_lvl'] = data['location'].astype(str).apply(lambda x: json.loads(x)['third_lvl'])

data.drop(['client_type', 'browser_source'], axis=1, inplace=True)
data['auth_type'].fillna('__NaN__', inplace=True)

for col in tqdm(['user_name', 'action', 'auth_type', 'ip',
                 'ip_location_type_keyword', 'ip_risk_level', 'location', 'device_model',
                 'os_type', 'os_version', 'browser_type', 'browser_version',
                 'bus_system_code', 'op_target', 'location_first_lvl', 'location_sec_lvl',
                 'location_third_lvl']):
    lbl = LabelEncoder()
    data[col] = lbl.fit_transform(data[col])

data['op_date'] = pd.to_datetime(data['op_date'])
data['op_ts'] = data["op_date"].values.astype(np.int64) // 10 ** 9
data = data.sort_values(by=['user_name', 'op_ts']).reset_index(drop=True)
data['last_ts'] = data.groupby(['user_name'])['op_ts'].shift(1)
data['ts_diff1'] = data['op_ts'] - data['last_ts']

for f in ['ip', 'location', 'device_model', 'os_version', 'browser_version']:
    data[f'user_{f}_nunique'] = data.groupby(['user_name'])[f].transform('nunique')

for method in ['mean', 'max', 'min', 'std']:
    data[f'ts_diff1_{method}'] = data.groupby('user_name')['ts_diff1'].transform(method)

train = data[data['risk_label'].notna()]
test = data[data['risk_label'].isna()]
train.fillna(-999, inplace=True)
test.fillna(-999, inplace=True)

y_col = 'risk_label'
feature_names = list(filter(lambda x: x not in [y_col, 'session_id', 'op_date', 'last_ts'], train.columns))

estimators = [
    ('lightgbm', get_base_model('lightgbm')),
    ('xgboost', get_base_model('xgboost')),
    ('catboost', get_base_model('catboost')),
    ('adaboost', get_base_model('adaboost')),
    ('random forest', get_base_model('random forest')),
    ('extra trees', get_base_model('extra trees')),
    ('svc', get_base_model('svc')),
    ('lr', get_base_model('lr')),
    ('knn', get_base_model('knn'))
]

model = StackingClassifier(estimators=estimators, final_estimator=get_base_model('lr'), n_jobs=32, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
model.fit(train[feature_names], train[y_col])

test_pred = model.predict_proba(test[feature_names])[:, 1]
train_pred = model.predict_proba(train[feature_names])[:, 1]
print('roc_auc_score', roc_auc_score(train[y_col], train_pred))
submit = pd.read_csv("submit.csv")
sns.distplot(submit['ret'])
plt.savefig('results.png')
submit["ret"] = submit["ret"].apply(lambda x: 1 if x >= 0.5 else 0)
submit.to_csv("submit.csv", index=False)
