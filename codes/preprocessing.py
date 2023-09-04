import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(data_path):
    train_df = pd.read_csv(os.path.join(data_path, 'train.csv'))
    building_info = pd.read_csv(os.path.join(data_path, 'building_info.csv'))
    test_df = pd.read_csv(os.path.join(data_path, 'test.csv'))

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
    train_df.drop('num_date_time', axis = 1, inplace=True)
    train_df.drop('sunshine', axis = 1, inplace=True)
    train_df.drop('solar_radiation', axis = 1, inplace=True)


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
    # building_info.drop('Unnamed: 0', axis = 1 , inplace=True)
    le = LabelEncoder()
    building_info['building_type'] = le.fit_transform(building_info['building_type'])

    building_info.drop('solar_power_capacity', axis = 1 , inplace=True)
    building_info.drop('ess_capacity', axis = 1 , inplace=True)
    building_info.drop('pcs_capacity', axis = 1 , inplace=True)
    ###

    test_df = test_df.rename(columns={
    '건물번호': 'building_number',
    '일시': 'date_time',
    '기온(C)': 'temperature',
    '강수량(mm)': 'rainfall',
    '풍속(m/s)': 'windspeed',
    '습도(%)': 'humidity',
    '전력소비량(kWh)': 'power_consumption'
    })
    test_df.drop('num_date_time', axis = 1, inplace=True)

    # Merge building_info with train and test data
    train_df = train_df.merge(building_info, on='building_number', how='left')
    test_df = test_df.merge(building_info, on='building_number', how='left')
    
    # Convert 'date_time' column to datetime format
    train_df['date_time'] = pd.to_datetime(train_df['date_time'])
    test_df['date_time'] = pd.to_datetime(test_df['date_time'])

    # Create additional time features
    for df in [train_df, test_df]:
        df['hour'] = df['date_time'].dt.hour
        df['day'] = df['date_time'].dt.day
        df['month'] = df['date_time'].dt.month
        df['day_of_year'] = df['date_time'].dt.dayofyear
        df['hour_of_year'] = df['date_time'].dt.hour + (df['date_time'].dt.dayofyear - 1) * 24
        df['day_of_week'] = df['date_time'].dt.dayofweek
        df['year'] = df['date_time'].dt.year
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)

    # Fill missing values and add interaction features
    for df in [train_df, test_df]:
        df['rainfall'].fillna(0, inplace=True)
        df['windspeed'].fillna(df['windspeed'].mean(), inplace=True)
        df['humidity'].fillna(df['humidity'].mean(), inplace=True)
        df['temp*humidity'] = df['temperature'] * df['humidity']
        

    for df in [train_df, test_df]:
        df['sin_time'] = np.sin(2 * np.pi * df.hour / 24)
        df['cos_time'] = np.cos(2 * np.pi * df.hour / 24)

    # THI 변수 추가
    # for df in [train_df, test_df]:
        # df['THI'] = (9/5 * df['temperature']) - 0.55 * (1 - df['humidity'] / 100) * (9/5 * df['humidity'] - 26) + 32

    # CDH 변수 추가
    def CDH(xs):
        ys = []
        for i in range(len(xs)):
            if i < 11:
                ys.append(np.sum(xs[:(i+1)]-26))
            else:
                ys.append(np.sum(xs[(i-11):(i+1)]-26))
        return np.array(ys)

    for num in range(1, 101, 1):
        for df in [train_df, test_df]:
            temp = df[df['building_number'] == num]
            cdh = CDH(temp['temperature'].values)
            df.loc[df['building_number'] == num, 'CDH'] = cdh

    ## Holiday 변수 추가
    holidays = [
        '2022-01-01', '2022-01-31', '2022-02-01', '2022-02-02', 
        '2022-03-01', '2022-03-09', '2022-05-05', '2022-05-08', 
        '2022-06-01', '2022-06-06', '2022-08-15', '2022-09-09', 
        '2022-09-10', '2022-09-11', '2022-09-12', '2022-10-03', 
        '2022-10-09', '2022-10-10', '2022-12-25'
    ]
    
    for df in [train_df, test_df]:
        df['is_holiday'] = df['date_time'].dt.date.astype('str').apply(lambda x: 1 if x in holidays else 0)

    for df in [train_df, test_df]: 
        # 1. 시간당 평균 전력 소비 피쳐
        hourly_avg = df.groupby('hour')['power_consumption'].transform('mean')
        df['hourly_avg_power'] = hourly_avg

        # 3. 건물 타입별 평균 전력 소비량 피쳐
        building_avg = df.groupby('building_number')['power_consumption'].transform('mean')
        df['building_avg_power'] = building_avg

        # 4. 특정 건물의 전력 소비량의 변동성 피쳐
        df['power_std'] = df.groupby('building_number')['power_consumption'].transform('std')

        # 5. 냉방면적 대비 전력 소비 피쳐
        df['power_per_cooling'] = df['power_consumption'] / df['cooling_area']

        # 6. Lagged Features
        for lag in range(1, 4):  # 3 hours lag
            df[f'lag_{lag}_hour'] = df.groupby('building_number')['power_consumption'].shift(lag)

        # 7. Rolling Statistics
        window_size = 3  # 3 hours window
        df['rolling_mean'] = df.groupby('building_number')['power_consumption'].rolling(window=window_size).mean().reset_index(0, drop=True)
        df['rolling_std'] = df.groupby('building_number')['power_consumption'].rolling(window=window_size).std().reset_index(0, drop=True)

        # 8. Seasonal Decomposition (일별 계절성 가정)
        result = seasonal_decompose(df['power_consumption'], model='additive', period=24)
        df['trend'] = result.trend
        df['seasonal'] = result.seasonal
        df['resid'] = result.resid

        # NaN 값 처리
        df.fillna(method='bfill', inplace=True)
        df.fillna(method='ffill', inplace=True)

    return train_df, test_df


if __name__ == "__main__":
    data_path = './data/'
    train_df, test_df = load_and_preprocess_data(data_path)
    train_df.to_csv(os.path.join(data_path,"preprocessed_train.csv"), index=False)
    test_df.to_csv(os.path.join(data_path,"preprocessed_test.csv"), index=False)