import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, AdaBoostClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def get_base_model(model_type=None):
    if model_type == 'xgboost':
        return xgb.XGBClassifier(objective='binary:logistic',
                                 boosting_type='dart',
                                 num_leaves=2 ** 8,
                                 max_depth=16,
                                 learning_rate=0.01,
                                 n_estimators=10000,
                                 subsample=1,
                                 colsample_bytree=0.8,
                                 reg_alpha=0.,
                                 reg_lambda=0.,
                                 scale_pos_weight=5,
                                 random_state=2048,
                                 n_jobs=32,
                                 metric='auc')
    elif model_type == 'lightgbm':
        return lgb.LGBMClassifier(objective='binary',
                                  boosting_type='dart',
                                  num_leaves=2 ** 8,
                                  max_depth=16,
                                  learning_rate=0.01,
                                  n_estimators=10000,
                                  subsample=1,
                                  colsample_bytree=0.8,
                                  reg_alpha=0.,
                                  reg_lambda=0.,
                                  random_state=2048,
                                  n_jobs=32,
                                  is_unbalance=True,
                                  metric='auc')
    elif model_type == 'random forest':
        return RandomForestClassifier(n_estimators=1000, n_jobs=32, random_state=42)
    elif model_type == "catboost":
        return CatBoostClassifier(n_estimators=1000, random_state=42, thread_count=32, custom_metric="AUC", early_stopping_rounds=50)
    elif model_type == "extra trees":
        return ExtraTreesClassifier(n_estimators=1000, n_jobs=32, random_state=42)
    elif model_type == "lr":
        return LogisticRegression(random_state=42)
    elif model_type == "svc":
        return SVC(random_state=42)
    elif model_type == "adaboost":
        return AdaBoostClassifier(n_estimators=1000, random_state=42)
    elif model_type == "knn":
        return KNeighborsClassifier(n_neighbors=5, n_jobs=32)
    elif model_type == "isolation forest":
        return IsolationForest(n_estimators=1000, random_state=42, n_jobs=32)
