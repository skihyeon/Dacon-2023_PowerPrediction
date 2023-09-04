import os
import sys
import xgboost as xgb
import pandas as pd
from util import TQDMProgressBar, smape, weighted_mse, custom_evals, save_to_csv

# Global Variables
PROJ_NAME = sys.argv[1] if len(sys.argv) > 1 else "NO PROJECT"

# Paths
train_data_path = "./data/preprocessed_train.csv"
model_save_dir = f"./saved_models/models_{PROJ_NAME}/"

def load_data_and_params():
    """Load training data and optimal parameters."""
    train_df = pd.read_csv(train_data_path)
    
    # Check if "parameters_opt.csv" exists, otherwise load "parameters.csv"
    if os.path.exists(f'./parameters/{PROJ_NAME}/parameters_opt.csv'):
        optimal_params_df = pd.read_csv(f'./parameters/{PROJ_NAME}/parameters_opt.csv')
    else:
        optimal_params_df = pd.read_csv(f'./parameters/{PROJ_NAME}/parameters.csv')
    
    return train_df, optimal_params_df

def extract_building_params(building_num, optimal_params_df):
    """Extract optimal parameters for a given building number."""
    building_params = optimal_params_df[optimal_params_df["building"] == building_num].iloc[0]
    extracted_params = {
        'colsample_bytree': building_params['colsample_bytree'],
        'learning_rate': building_params['learning_rate'],
        'max_depth': int(building_params['max_depth']),
        'alpha': building_params['alpha'],
        'min_child_weight': int(building_params['min_child_weight']),
        'subsample': building_params['subsample'],
        'wmse_weight' : building_params['wmse_weight']
    }
    return extracted_params, int(building_params['num_round'])


def train_with_optimal_params(train_df, building_num, params, num_round):
    """Train a model for a given building using optimal parameters."""
    # Filter data for the current building
    building_data = train_df[train_df.building_number == building_num]
    X = building_data.drop(['building_number', 'date_time', 'power_consumption'], axis=1, errors='ignore')
    y = building_data['power_consumption']

    dtrain = xgb.DMatrix(X, label=y, feature_names=X.columns)
    xgb_model = xgb.train(params, 
                          dtrain, 
                          num_round, 
                          obj=weighted_mse(params['wmse_weight']),
                          custom_metric=custom_evals,
                          callbacks=[TQDMProgressBar(num_round)],   
                          maximize=False,
                          verbose_eval=num_round)
    
    # Save the model
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    model_path = f"{model_save_dir}/xgb_model_building_{building_num}.bin"
    xgb_model.save_model(model_path)

    print(f"Model for building {building_num} saved to {model_path}")


def main():
    """Main execution function."""
    train_df, optimal_params_df = load_data_and_params()
    
    for building_num in range(1, 101):
        params, num_round = extract_building_params(building_num, optimal_params_df)
        train_with_optimal_params(train_df, building_num, params, num_round)


if __name__ == "__main__":
    main()