import os
import xgboost as xgb
from tqdm import tqdm
import pandas as pd
import numpy as np


class TQDMProgressBar(xgb.callback.TrainingCallback):
    def __init__(self, num_round):
        self.pbar = None
        self.num_round = num_round

    def before_training(self, model):
        self.pbar = tqdm(total=self.num_round, desc="Training", position=0, leave=True)
        return model

    def after_iteration(self, model, epoch, evals_log):
        self.pbar.update(1)
        return False

    def after_training(self, model):
        self.pbar.close()
        return model

def smape(y_true, y_pred):
    """Calculate Symmetric Mean Absolute Percentage Error"""
    return 100 * (2 * abs(y_pred - y_true) / (abs(y_true) + abs(y_pred))).mean()

def weighted_mse(alpha=1):
    def weighted_mse_fixed(pred, dtrain): 
        label = dtrain.get_label()
        residual = (label - pred).astype("float")
        grad = np.where(residual > 0, -2 * alpha * residual, -2 * residual)
        hess = np.where(residual > 0, 2 * alpha, 2.0)
        return grad, hess
    return weighted_mse_fixed

def custom_evals(preds, dtrain):
    labels = dtrain.get_label()
    return [('smape', smape(labels, preds))]

def save_to_csv(file_name, config, smape_value, run_time, building_num, best_iteration):
    """Helper function to save config and smape_value to CSV."""
    dir_path = f"../results/{run_time}/"
    filename = os.path.join(dir_path, f"sweep_results_{file_name}.csv")
    extracted_config = {key: config[key] for key in sweep_config['parameters'].keys()}
    extracted_config["best_iteration"] = best_iteration
    extracted_config["smape"] = smape_value
    extracted_config["building"] = building_num
    new_df = pd.DataFrame([extracted_config])
    column_order = ["building"] + list(sweep_config['parameters'].keys()) + ["best_iteration", "smape"]
    new_df = new_df[column_order]
    if not os.path.exists(filename):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        new_df.to_csv(filename, index=False)
    else:
        existing_df = pd.read_csv(filename)
        final_df = pd.concat([existing_df, new_df], axis=0)
        final_df.to_csv(filename, index=False)
