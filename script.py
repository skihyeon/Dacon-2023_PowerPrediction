import subprocess
import argparse

def preprocess_data():
    subprocess.run(["python", "./codes/preprocessing.py"])

def train_model(PROJ_NAME):
    subprocess.run(["python", "./codes/model_tunning.py", PROJ_NAME])

def inference_model(PROJ_NAME):
    subprocess.run(["python", "./codes/inference.py", PROJ_NAME])

def run_bayesian_optimization(PROJ_NAME):
    subprocess.run(["python", "./codes/bayesian_optimization.py", PROJ_NAME])

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run parts of the project pipeline.")
    parser.add_argument('-p', '--preprocess', action='store_true', help="Run data preprocessing.")
    parser.add_argument('-t', '--train', action='store_true', help="Run model training.")
    parser.add_argument('-i', '--inference', action='store_true', help="Run model inference.")
    parser.add_argument('-b', '--bayesian', action='store_true', help="Run Bayesian optimization.")
    parser.add_argument('proj_name', type=str, nargs='?', default="NO_PROJECT", help="Project name to be used for saving or loading models.")

    args = parser.parse_args()

    if args.preprocess:
        preprocess_data()
    if args.bayesian:
        run_bayesian_optimization(args.proj_name)
    if args.train:
        train_model(args.proj_name)
    if args.inference:
        inference_model(args.proj_name)
    
