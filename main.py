#!/usr/bin/env python
# coding: utf-8

import warnings

warnings.simplefilter('ignore')

import json
import numpy as np
import pandas as pd
import gc
from tqdm import tqdm

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
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

isolation_forest = get_base_model("isolation forest")
isolation_forest.fit(train[feature_names], train[y_col])

train['iso_pred'] = isolation_forest.predict(train[feature_names])
test['iso_pred'] = isolation_forest.predict(test[feature_names])

feature_names.append('iso_pred')

model = get_base_model('lightgbm')
oof = []
prediction = test[['session_id']]
prediction[y_col] = 0
df_importance_list = []

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=2048)
for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(train[feature_names], train[y_col])):
    X_train = train.iloc[trn_idx][feature_names]
    Y_train = train.iloc[trn_idx][y_col]
    X_val = train.iloc[val_idx][feature_names]
    Y_val = train.iloc[val_idx][y_col]
    print('\nFold_{} Training ================================\n'.format(fold_id + 1))

    lgb_model = model.fit(X_train,
                          Y_train,
                          eval_names=['train', 'valid'],
                          eval_set=[(X_train, Y_train), (X_val, Y_val)],
                          verbose=500,
                          eval_metric='auc',
                          early_stopping_rounds=50)

    pred_val = lgb_model.predict_proba(X_val, num_iteration=lgb_model.best_iteration_)
    df_oof = train.iloc[val_idx][['session_id', y_col]].copy()
    df_oof['pred'] = pred_val[:, 1]
    oof.append(df_oof)

    pred_test = lgb_model.predict_proba(
        test[feature_names], num_iteration=lgb_model.best_iteration_)
    prediction[y_col] += pred_test[:, 1] / kfold.n_splits

    df_importance = pd.DataFrame({
        'column': feature_names,
        'importance': lgb_model.feature_importances_,
    })
    df_importance_list.append(df_importance)

    del lgb_model, pred_val, pred_test, X_train, Y_train, X_val, Y_val
    gc.collect()

df_importance = pd.concat(df_importance_list)
df_importance = df_importance.groupby(['column'])['importance'].agg('mean').sort_values(ascending=False).reset_index()
print(df_importance)

df_oof = pd.concat(oof)
print('roc_auc_score', roc_auc_score(df_oof[y_col], df_oof['pred']))
prediction['id'] = range(len(prediction))
prediction['id'] = prediction['id'] + 1
prediction = prediction[['id', 'risk_label']].copy()
prediction.columns = ['id', 'ret']
print(prediction.head())
prediction.to_csv("submit.csv", index=False)
