import numpy as np
import pandas as pd
import argparse

import os  

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from azureml.core.run import Run
# from azureml.data.dataset_factory import TabularDatasetFactory
import joblib

def clean_data(data):
      # Create a copy of the original dataframe
    heart_clean_df = data.copy()

    # Impute 0 values in RestingBP with median value of the column grouped by HeartDisease
    #heart_clean_df['RestingBP'] = heart_clean_df.groupby('HeartDisease')['RestingBP'].apply(lambda x: x.replace(0, x.median()))

    # Impute 0 values in Cholesterol with median value of the column grouped by HeartDisease
    #heart_clean_df['Cholesterol'] = heart_clean_df.groupby('HeartDisease')['Cholesterol'].apply(lambda x: x.replace(0, x.median()))

    # Convert categorical variable into dummy variables
    heart_clean_df = pd.get_dummies(heart_clean_df, drop_first=True)
    
    print(heart_clean_df.columns)
             
    # Split data into features 'X' and target variable 'y'
    X = heart_clean_df#.drop('HeartDisease', axis=1)
    #X = heart_clean_df
    y = heart_clean_df['HeartDisease']
    
    return X, y


def main():
    
    # Add arguments to script
    parser = argparse.ArgumentParser()

    parser.add_argument('--n_neighbors', type=int, default=5, help="The number of neighbors to consider")
    parser.add_argument('--weights', type=str, default='uniform', help="Maximum number of iterations to converge")
    parser.add_argument('--metric', type=str, default='minkowski', help="The distance metric used for the tree.")
    
    args = parser.parse_args()
    #
    run = Run.get_context()
    
    heart_disease_df = pd.read_csv('data.csv')
    #heart_disease_df = pd.read_csv('heart_disease_prediction.csv')
    
    # Clean data
    X, y = clean_data(heart_disease_df)
    
    # Create list of selected features
    sel_features = [
                    'Oldpeak',
                    'Sex_M',
                    'ExerciseAngina',
                    'ST_Slope_Flat',
                    'ST_Slope_Up'
    ]
    
    # Split data for training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X[sel_features], y,
                                                      test_size=0.15,
                                                      random_state=0)

    # Instantiate scaler
    scaler = MinMaxScaler()

    # Fit and transform selected features in the training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform same features in the validation set
    X_val_scaled = scaler.transform(X_val)

    # Fit the model on scaled features (using default n_neighbors value)
    knn = KNeighborsClassifier(n_neighbors=args.n_neighbors, weights=args.weights, metric=args.metric)
    knn.fit(X_train_scaled, y_train)

    # Evaluate the model on scaled features
    accuracy = knn.score(X_val_scaled, y_val)

    print(f'Model accuracy: {accuracy*100:.2f}%')
    run.log("Accuracy", float(accuracy))
    
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(knn, 'outputs/model.joblib')

if __name__ == '__main__':
    main()
