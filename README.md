# Project Title
2023 Power Prediction Competition (auspice. Korea Energy Agency)

# Project Overview
The goal of this competition, hosted by the Korea Energy Agency, is to discover efficient artificial intelligence algorithms through electricity consumption prediction simulations to ensure stable and efficient energy supply. Accurate predictions of power consumption are essential for achieving a reliable and efficient energy supply.

# Competition Participation
- Rank
  - public : 37/2020
  - private : 44/2020

## Tech Stack
- Python
- pandas
- scikit-learn
- XGBoost

## Data Sources
The project uses the following data sources:
- `train.csv`: Training data
- `test.csv`: Test data
- `building_info.csv`: Building information data
- if you want to get datas, please visit (https://dacon.io/competitions/official/236125/overview/description)

## Data Preprocessing
- Load and preprocess training and test data.
- Translate and label encode building information data.
- Handle date and time data and create various time-related features.
- Handle missing values and add interaction features.

## Model Training
- Train XGBoost models with different optimal hyperparameters for each building.
- Optimal hyperparameters for each building are extracted from the `parameters_opt.csv` file.
- Models are trained using XGBoost, aiming to minimize the weighted mean squared error.

## Evaluation Metrics
The project uses the following evaluation metrics:
- SMAPE (Symmetric Mean Absolute Percentage Error)
- Weighted Mean Squared Error


