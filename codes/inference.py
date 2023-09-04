import sys
import xgboost as xgb
import pandas as pd
from datetime import datetime
from tqdm import tqdm 

PROJ_NAME = sys.argv[1] if len(sys.argv) > 1 else "NO PROJECT"

def load_model_for_building(building_num):
    """Load the model for a specific building number."""
    model_path = f"./saved_models/models_{PROJ_NAME}/xgb_model_building_{building_num}.bin"
    loaded_model = xgb.Booster()
    loaded_model.load_model(model_path)
    return loaded_model

def predict_for_building(building_data):
    """Predict power consumption for a specific building."""
    X_test = building_data.drop(['building_number', 'date_time'], axis=1, errors='ignore')
    building_num = building_data['building_number'].iloc[0]

    loaded_model = load_model_for_building(building_num)
    dtest = xgb.DMatrix(X_test)
    preds = loaded_model.predict(dtest)
    return preds

def save_submission(all_preds):
    """Save the predictions to a submission file."""
    submission = pd.read_csv('./submissions/sample_submission.csv')
    submission['answer'] = all_preds
    submission.to_csv(f'./submissions/{PROJ_NAME}_submission.csv', index=False)
    print(f"Submission saved to: ./submissions/{PROJ_NAME}_submission.csv")

def main():
    """Main execution function."""
    test_df = pd.read_csv("./data/preprocessed_test.csv")
    all_preds = []

    for building_num in tqdm(range(1, 101), desc="buildings"):
        building_data = test_df[test_df.building_number == building_num]
        preds = predict_for_building(building_data)
        all_preds.extend(preds)

    save_submission(all_preds)

if __name__ == "__main__":
    main()

