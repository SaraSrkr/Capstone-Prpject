import argparse
import joblib
import os
from azureml.core import Run
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from azureml.core.run import Run

def main():
    # Add arguments to script
    parser = argparse.ArgumentParser()
    parser.add_argument('--C', type=float, default=1.0, help="Inverse of regularization strength. Smaller values cause stronger regularization")
    parser.add_argument('--max_iter', type=int, default=100, help="Maximum number of iterations to converge")    parser.add_argument("--input-data", type=str, dest='input_data', help='training dataset')
    
    args = parser.parse_args()
    
    run = Run.get_context()

    diabetes = run.input_datasets['training_data'].to_pandas_dataframe() # Get the training data from the estimator input
    
    X, y = diabetes[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']].values, diabetes['Outcome'].values


    run.log("Regularization Strength:", np.float(args.C))
    run.log("Max iterations:", np.int(args.max_iter))
    

    # TODO: Split data into train and test sets.

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    model = LogisticRegression(C=args.C, max_iter=args.max_iter).fit(x_train, y_train)

   

    AUC_weighted = model.score(x_test, y_test)
    run.log("AUC_weighted", np.float(AUC_weighted))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(model, 'outputs/hyperdrivemodel.joblib')

if __name__ == '__main__':
   
    
    main()
