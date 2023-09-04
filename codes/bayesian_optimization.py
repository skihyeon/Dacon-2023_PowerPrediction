import os
import numpy as np
import xgboost as xgb
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split
from tqdm import tqdm
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from util import TQDMProgressBar, smape, weighted_mse, custom_evals, save_to_csv
import sys

PROJ_NAME = sys.argv[1] if len(sys.argv) > 1 else "NO_PROJECT"

NUM_ROUNDS = 50000
PARAMETER_SPACE = {
    'learning_rate': hp.choice('learning_rate', [0.01]),
    'min_child_weight': hp.choice('min_child_weight', list(range(7, 11))),
    'max_depth': hp.choice('max_depth', list(range(7, 11))),
    'colsample_bytree': hp.choice('colsample_bytree', [0.8, 0.9]),
    'subsample': hp.choice('subsample', [0.8, 0.9]),
    'alpha': hp.choice('alpha', [0, 0.001, 0.01, 1, 10, 100]),
    'wmse_weight': hp.choice('wmse_weight', [1, 10, 25, 50, 75, 100])
}

def evaluate_model(params, dtrain, dtest):
    """Train and evaluate the XGBoost model."""
    wmse_weight = params.pop('wmse_weight', None)  # Extract and remove wmse_weight
    xgb_model = xgb.train(
        params,
        dtrain,
        NUM_ROUNDS,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        obj=weighted_mse(wmse_weight),  # Pass wmse_weight here
        early_stopping_rounds=20,
        verbose_eval=False
    )
    y_pred = xgb_model.predict(dtest)
    best_iteration = xgb_model.best_iteration
    params['wmse_weight'] = wmse_weight  # Add wmse_weight back to params
    return smape(dtest.get_label(), y_pred), xgb_model, best_iteration, params



def objective(params, dtrain, dtest):
    """Objective function for hyperopt."""
    smape_value, _, _, _ = evaluate_model(params, dtrain, dtest)
    return {'loss': smape_value, 'status': STATUS_OK}

def main():
    train = pd.read_csv('./data/preprocessed_train.csv')
    
    # 1. results_df 초기화 및 CSV 파일 저장
    columns = ['building', 'num_round', 'learning_rate', 'min_child_weight', 'max_depth',
               'colsample_bytree', 'subsample', 'alpha', 'wmse_weight', 'smape']
    results_df = pd.DataFrame(columns=columns)
    
    directory = f'./parameters/{PROJ_NAME}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    csv_path = f'{directory}/parameters_opt.csv'
    results_df.to_csv(csv_path, index=False)

    for building_num in tqdm(range(1, 101), desc="Buildings"):
        y = train.loc[train.building_number == building_num, 'power_consumption']
        x = train.loc[train.building_number == building_num].iloc[:, 3:]
        y_train, y_test, x_train, x_test = temporal_train_test_split(y=y, X=x, test_size=168)
        dtrain = xgb.DMatrix(x_train, label=y_train)
        dtest = xgb.DMatrix(x_test, label=y_test)

        best = fmin(fn=lambda params: objective(params, dtrain, dtest), space=PARAMETER_SPACE, algo=tpe.suggest, max_evals=20)
        best_params = space_eval(PARAMETER_SPACE, best)
        smape_value, best_model, best_iteration, final_params = evaluate_model(best_params, dtrain, dtest)

        # 2. 각 건물에 대한 베이지안 최적화 결과를 results_df에 추가
        result_row = {
            'building': building_num,
            'num_round': best_iteration,
            **final_params,
            'smape': smape_value
        }
        results_df = results_df._append(result_row, ignore_index=True)
        
        # 3. results_df를 사용하여 CSV 파일 업데이트
        results_df.to_csv(csv_path, index=False)
if __name__ == "__main__":
    main()