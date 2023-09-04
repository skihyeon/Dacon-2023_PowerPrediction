import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import model_tunning
import inference

# 1. 데이터 전처리 함수
def modified_preprocessing(train_df, test_df):
    building_info = pd.read_csv("/mnt/data/building_info.csv")
    
    train_df = train_df.rename(columns={
        '건물번호': 'building_number',
        '일시': 'date_time',
        '기온(C)': 'temperature',
        '강수량(mm)': 'rainfall',
        '풍속(m/s)': 'windspeed',
        '습도(%)': 'humidity',
        '일조(hr)': 'sunshine',
        '일사(MJ/m2)': 'solar_radiation',
        '전력소비량(kWh)': 'power_consumption'
    })

    building_info = building_info.rename(columns={
        '건물번호': 'building_number',
        '건물유형': 'building_type',
        '연면적(m2)': 'total_area',
        '냉방면적(m2)': 'cooling_area',
        '태양광용량(kW)': 'solar_power_capacity',
        'ESS저장용량(kWh)': 'ess_capacity',
        'PCS용량(kW)': 'pcs_capacity'
    })

    translation_dict = {
        '건물기타': 'Other Buildings',
        '공공': 'Public',
        '대학교': 'University',
        '데이터센터': 'Data Center',
        '백화점및아울렛': 'Department Store and Outlet',
        '병원': 'Hospital',
        '상용': 'Commercial',
        '아파트': 'Apartment',
        '연구소': 'Research Institute',
        '지식산업센터': 'Knowledge Industry Center',
        '할인마트': 'Discount Mart',
        '호텔및리조트': 'Hotel and Resort'
    }

    building_info['building_type'] = building_info['building_type'].replace(translation_dict)
    le = LabelEncoder()
    building_info['building_type'] = le.fit_transform(building_info['building_type'])

    train_df = pd.merge(train_df, building_info, how='left', on='building_number')
    test_df = pd.merge(test_df, building_info, how='left', on='building_number')

    # Additional preprocessing steps (if needed)...

    return train_df, test_df

# 2. 학습 및 추론 함수
def train_and_infer(train_data, test_data, features_to_remove=[], features_to_add=[]):
    preprocessed_train, preprocessed_test = modified_preprocessing(train_data, test_data)
    
    # Remove or add features based on arguments
    preprocessed_train = preprocessed_train.drop(columns=features_to_remove)
    preprocessed_test = preprocessed_test.drop(columns=features_to_remove)
    # Add features if needed...

    model = model_tunning.train(preprocessed_train)
    predictions = inference.predict(preprocessed_test, model)
    
    return predictions

# 3. 성능 평가 함수
def evaluate_performance(base_submission_path, predictions):
    base_submission = pd.read_csv(base_submission_path)
    rmse = np.sqrt(np.mean((base_submission['전력사용량(kWh)'] - predictions) ** 2))
    return rmse

# 결과와 피처 저장 함수
def save_results_and_features(base_score, new_score, base_features, modified_features):
    # Save results
    result_df = pd.DataFrame({
        "RMSE": [base_score, new_score],
        "Description": ["Base Features", "Modified Features"]
    })
    result_df.to_csv("/mnt/data/evaluation_results.csv", index=False)

    # Save features
    feature_df = pd.DataFrame({
        "Base Features": base_features,
        "Modified Features": modified_features + [""] * (len(base_features) - len(modified_features))
    })
    feature_df.to_csv("/mnt/data/used_features.csv", index=False)

# Main Execution
if __name__ == "__main__":
    train = pd.read_csv("/mnt/data/train.csv")
    test = pd.read_csv("/mnt/data/test.csv")
    
    # With Base Features
    base_predictions, base_features = train_and_infer(train, test)
    base_score = evaluate_performance("/mnt/data/20230818_1908_submission.csv", base_predictions)

    # With Modified Features (example: remove 'month')
    modified_predictions, modified_features = train_and_infer(train, test, features_to_remove=['month'])
    modified_score = evaluate_performance("/mnt/data/20230818_1908_submission.csv", modified_predictions)

    # Save results and features to CSV
    save_results_and_features(base_score, modified_score, base_features, modified_features)