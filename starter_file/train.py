import numpy as np
import pandas as pd

import os  

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from azureml.core.run import Run
from azureml.core import Workspace
from azureml.data.dataset_factory import TabularDatasetFactory

def clean_data(data):
      # Create a copy of the original dataframe
    heart_clean_df = data.copy()

    # Impute 0 values in RestingBP with median value of the column grouped by HeartDisease
    heart_clean_df['RestingBP'] = heart_clean_df.groupby('HeartDisease')['RestingBP'].apply(lambda x: x.replace(0, x.median()))

    # Impute 0 values in Cholesterol with median value of the column grouped by HeartDisease
    heart_clean_df['Cholesterol'] = heart_clean_df.groupby('HeartDisease')['Cholesterol'].apply(lambda x: x.replace(0, x.median()))

    # Convert categorical variable into dummy variables
    heart_clean_df = pd.get_dummies(heart_clean_df, drop_first=True)
             
    # Split data into features 'X' and target variable 'y'
    # X = heart_clean_df#.drop('HeartDisease', axis=1)
    X = heart_clean_df
    y = heart_clean_df['HeartDisease']
    
    return X, y


def main():
    ws1 = Workspace.from_config()
    
    #
    run = Run.get_context()
    
        # Try to load the dataset from the Workspace. Otherwise, create it from the file
    # NOTE: update the key to match the dataset name
    found = False
    key = "HeartFailurePrediction"
    description_text = "Heart Failure Prediction Dataset Kaggle"

    if key in ws1.datasets.keys(): 
            found = True
            dataset = ws1.datasets[key] 

    if not found:      
            # Set Kaggle credentials as environment variables or in a separate Kaggle.json file  
            os.environ['KAGGLE_USERNAME'] = '<your_kaggle_username>'  
            os.environ['KAGGLE_KEY'] = '<your_kaggle_api_key>'  

            # kaggle datasets download -d fedesoriano/heart-failure-prediction

            # Download the Kaggle dataset using the KaggleDatasetFactory  
            from azureml.data.dataset_factory import KaggleDatasetFactory  

            factory = KaggleDatasetFactory()  
            dataset = factory.create_dataset(  
                dataset_id='<kaggle_dataset_id>',  
                version='<kaggle_dataset_version>'  
            )  

            #Register Dataset in Workspace
            dataset = dataset.register(workspace=ws1pa,
                                       name=key,
                                       description=description_text)
   
    # Convert to Pandas DF
    heart_disease_df = dataset.to_pandas_dataframe()
    
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
                                                      random_state=417)

    # Instantiate scaler
    scaler = MinMaxScaler()

    # Fit and transform selected features in the training set
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform same features in the validation set
    X_val_scaled = scaler.transform(X_val)

    # Fit the model on scaled features (using default n_neighbors value)
    knn = KNeighborsClassifier()
    knn.fit(X_train_scaled, y_train)

    # Evaluate the model on scaled features
    accuracy = knn.score(X_val_scaled, y_val)

    print(f'Model accuracy: {accuracy*100:.2f}%')
    #run.log("Accuracy", np.float(accuracy))

if __name__ == '__main__':
    main()
