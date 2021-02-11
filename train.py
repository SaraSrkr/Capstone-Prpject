import argparse
import joblib
import os
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from azureml.core.run import Run
from azureml.data.dataset_factory import TabularDatasetFactory

url_bath=('https://raw.githubusercontent.com/SaraSrkr/Capstone-Prpject/main/diabetes.csv')
trainig_data =TabularDatasetFactory.from_delimited_files(url_bath)

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")   

    
    args = parser.parse_args()
    
    run = Run.get_context()

    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    

     # Get the training data

    url_bath=('https://raw.githubusercontent.com/SaraSrkr/Capstone-Prpject/main/diabetes.csv')
    trainig_data =TabularDatasetFactory.from_delimited_files(url_bath)
    diabetes = trainig_data.to_pandas_dataframe()

    x = diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values
    y = diabetes['Outcome'].values


    # TODO: Split data into train and test sets.

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

   

    accuracy = model.score(x_test, y_test)
    run.log("Accuracy", np.float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperdrivemodel.joblib')

if __name__ == '__main__':
   
    
    main()
