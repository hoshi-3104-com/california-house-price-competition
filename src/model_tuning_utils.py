import numpy as np
import pandas as pd
import optuna
import json
import os
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error

# 各モデルのインポート（未インストールの場合は事前に要pip install）
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.linear_model import Ridge

# Optunaの標準ログを抑制
optuna.logging.set_verbosity(optuna.logging.WARNING)

# =====================================================================
# 1. LightGBM用 チューニング関数
# =====================================================================
def tune_lgbm_parameters(X, y, stratify_bin, save_dir, filename='best_params_lgbm.json', n_trials=30, random_state=42):
    def objective(trial):
        params = {
            'random_state': random_state,
            'verbose': -1,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'num_leaves': trial.suggest_int('num_leaves', 31, 127),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
            'lambda_l1': trial.suggest_float('lambda_l1', 1e-5, 5.0, log=True),
            'lambda_l2': trial.suggest_float('lambda_l2', 1e-5, 5.0, log=True),
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X))
        for cv_train_idx, cv_val_idx in skf.split(X, stratify_bin):
            model = LGBMRegressor(**params)
            model.fit(X.iloc[cv_train_idx], y.iloc[cv_train_idx])
            oof_preds[cv_val_idx] = np.clip(model.predict(X.iloc[cv_val_idx]), 0, 5.00001)
        
        trial_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        tqdm.write(f"★ LGBM Trial {trial.number + 1:2d} / {n_trials} 完了 | 検証RMSE: {trial_rmse:.4f}")
        return trial_rmse

    print(f"--- OptunaによるLightGBMチューニングを開始 (試行回数: {n_trials}) ---")
    study = optuna.create_study(direction='minimize')
    pbar = tqdm(range(n_trials), desc="LGBM Tuning")
    for _ in pbar:
        study.optimize(objective, n_trials=1, n_jobs=1)
        pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
    
    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['verbose'] = -1
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✅ パラメータを保存しました: {os.path.join(save_dir, filename)}\n")
    return best_params


# =====================================================================
# 2. XGBoost用 チューニング関数
# =====================================================================
def tune_xgboost_parameters(X, y, stratify_bin, save_dir, filename='best_params_xgb.json', n_trials=30, random_state=42):
    def objective(trial):
        params = {
            'random_state': random_state,
            'verbosity': 0,
            'tree_method': 'hist',
            'enable_categorical': True,
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'max_depth': trial.suggest_int('max_depth', 4, 10),
            'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 20.0),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'alpha': trial.suggest_float('alpha', 1e-5, 5.0, log=True),       # L1正則化
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 5.0, log=True) # L2正則化
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X))
        for cv_train_idx, cv_val_idx in skf.split(X, stratify_bin):
            model = XGBRegressor(**params)
            model.fit(X.iloc[cv_train_idx], y.iloc[cv_train_idx])
            oof_preds[cv_val_idx] = np.clip(model.predict(X.iloc[cv_val_idx]), 0, 5.00001)
        
        trial_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        tqdm.write(f"★ XGBoost Trial {trial.number + 1:2d} / {n_trials} 完了 | 検証RMSE: {trial_rmse:.4f}")
        return trial_rmse

    print(f"--- OptunaによるXGBoostチューニングを開始 (試行回数: {n_trials}) ---")
    study = optuna.create_study(direction='minimize')
    pbar = tqdm(range(n_trials), desc="XGBoost Tuning")
    for _ in pbar:
        study.optimize(objective, n_trials=1, n_jobs=1)
        pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
        
    best_params = study.best_params
    best_params['random_state'] = random_state
    best_params['verbosity'] = 0
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✅ パラメータを保存しました: {os.path.join(save_dir, filename)}\n")
    return best_params


# =====================================================================
# 3. CatBoost用 チューニング関数
# =====================================================================
def tune_catboost_parameters(X, y, stratify_bin, save_dir, filename='best_params_cat.json', n_trials=20, random_state=42):
    # ※CatBoostは1回の学習が重いため、デフォルトのn_trialsを少し控えめの20にしています
    def objective(trial):
        params = {
            'random_seed': random_state,
            'verbose': 0,
            'iterations': trial.suggest_int('iterations', 100, 500),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0),
            'random_strength': trial.suggest_float('random_strength', 1e-3, 10.0, log=True),
            'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0)
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X))
        for cv_train_idx, cv_val_idx in skf.split(X, stratify_bin):
            model = CatBoostRegressor(**params)
            model.fit(X.iloc[cv_train_idx], y.iloc[cv_train_idx])
            oof_preds[cv_val_idx] = np.clip(model.predict(X.iloc[cv_val_idx]), 0, 5.00001)
            
        trial_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        tqdm.write(f"★ CatBoost Trial {trial.number + 1:2d} / {n_trials} 完了 | 検証RMSE: {trial_rmse:.4f}")
        return trial_rmse

    print(f"--- OptunaによるCatBoostチューニングを開始 (試行回数: {n_trials}) ---")
    study = optuna.create_study(direction='minimize')
    pbar = tqdm(range(n_trials), desc="CatBoost Tuning")
    for _ in pbar:
        study.optimize(objective, n_trials=1, n_jobs=1)
        pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
        
    best_params = study.best_params
    best_params['random_seed'] = random_state
    best_params['verbose'] = 0
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✅ パラメータを保存しました: {os.path.join(save_dir, filename)}\n")
    return best_params


# =====================================================================
# 4. Ridge回帰用 チューニング関数（線形モデル・超親切設計）
# =====================================================================
def tune_ridge_parameters(X, y, stratify_bin, save_dir, filename='best_params_ridge.json', n_trials=20, random_state=42):
    
    # 【GM設計】Ridgeは文字列やCategory型の変数を学習できないため、内部で自動的にダミー変数化（One-Hot）を施す
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    def objective(trial):
        params = {
            'alpha': trial.suggest_float('alpha', 1e-3, 100.0, log=True),  # 正則化の強さ
            'random_state': random_state
        }
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        oof_preds = np.zeros(len(X_encoded))
        for cv_train_idx, cv_val_idx in skf.split(X_encoded, stratify_bin):
            model = Ridge(**params)
            model.fit(X_encoded.iloc[cv_train_idx], y.iloc[cv_train_idx])
            oof_preds[cv_val_idx] = np.clip(model.predict(X_encoded.iloc[cv_val_idx]), 0, 5.00001)
            
        trial_rmse = np.sqrt(mean_squared_error(y, oof_preds))
        tqdm.write(f"★ Ridge Trial {trial.number + 1:2d} / {n_trials} 完了 | 検証RMSE: {trial_rmse:.4f}")
        return trial_rmse

    print(f"--- OptunaによるRidge回帰チューニングを開始 (試行回数: {n_trials}) ---")
    study = optuna.create_study(direction='minimize')
    pbar = tqdm(range(n_trials), desc="Ridge Tuning")
    for _ in pbar:
        study.optimize(objective, n_trials=1, n_jobs=1)
        pbar.set_postfix({"Best RMSE": f"{study.best_value:.4f}"})
        
    best_params = study.best_params
    best_params['random_state'] = random_state
    
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, filename), 'w') as f:
        json.dump(best_params, f, indent=4)
    print(f"✅ パラメータを保存しました: {os.path.join(save_dir, filename)}\n")
    return best_params