# Heart Failure Prediction
Cardiovascular diseases (CVDs) are the number 1 cause of death globally, taking an estimated 17.9 million lives each year, which accounts for 31% of all deaths worldwide. Four out of 5CVD deaths are due to heart attacks and strokes, and one-third of these deaths occur prematurely in people under 70 years of age. Heart failure is a common event caused by CVDs and this dataset contains 11 features that can be used to predict a possible heart disease.

People with cardiovascular disease or who are at high cardiovascular risk (due to the presence of one or more risk factors such as hypertension, diabetes, hyperlipidaemia or already established disease) need early detection and management wherein a machine learning model can be of great help.


## Project Set Up and Installation
The project utilized Udacity provided setup and preconfigured workspace.

## Dataset

### Overview
The dataset used for the project is the [Heart Failure Prediction](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data) dataset from Kaggle.The anonymized data was collected from multiple hospitals on several patients to predict the likelihood of a new patient having heart disease in the future. The dataset includes relevant information for each patient, such as personal information and medical data, including whether or not they have had heart disease before.

Attribute Information
* Age: age of the patient [years]
* Sex: sex of the patient [M: Male, F: Female]
* ChestPainType: chest pain type [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
* RestingBP: resting blood pressure [mm Hg]
* Cholesterol: serum cholesterol [mm/dl]
* FastingBS: fasting blood sugar [1: if FastingBS > 120 mg/dl, 0: otherwise]
* RestingECG: resting electrocardiogram results [Normal: Normal, ST: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), LVH: showing probable or definite left ventricular hypertrophy by Estes' criteria]
* MaxHR: maximum heart rate achieved [Numeric value between 60 and 202]
* ExerciseAngina: exercise-induced angina [Y: Yes, N: No]
* Oldpeak: oldpeak = ST [Numeric value measured in depression]
* ST_Slope: the slope of the peak exercise ST segment [Up: upsloping, Flat: flat, Down: downsloping]
* HeartDisease: output class [1: heart disease, 0: Normal]

### Task
We will build a model to accurately predict the likelihood of a new patient having heart disease in the future.

### Access
The dataset was created in AzureML studio by uploading the local file. The code to utilize KaggleDatasetFactory is included in the Notebook.
![Heart Failure Prediction Dataset](./screenshots/Dataset.png)

## Automated ML
- AutoML config utilized the compute cluster that was created to perform classification utilizing the dataset with 'HeartDisease' as the column to predict. 
- The experiment timeout was set to 20 mins with max concurrent iterations set to 5 to best utilize the compute cluster.
- The primary metric was accuracy.

### Results

#### AutoML Run In Progress
![AutoML Run Details In Progress](./screenshots/AutoML%20Run%20Details%20Start.png)

#### AutoML Run Complete
![AutoML Run Details Complete](./screenshots/AutoML%20Run%20Details%20Complete.png)

#### Best Model 
![Best Model Azure ML Studio](./screenshots/AutoML%20Studio.png)

## Hyperparameter Tuning
K-Nearest Neighbors Classifier was used with HeartFailurePrediction dataset. The classifier assigns the object to the class most common among its k nearest neighbors (k is a positive integer). Default values are as follow:
- n_neighbors: 5
- weights: 'uniform'
- algorithm: 'auto'
- leaf_size: 30
- p: 2
- metric: 'minkowski'
- metric_params: None
- n_jobs: None

The parameters used for Hypertuning
- '--n_neighbors': choice(7, 9, 12),  
- '--weights': choice('uniform', 'distance'),  
- '--metric': choice('euclidean', 'manhattan', 'chebyshev', 'minkowski')  

The BanditPolicy was used as the early termination policy. It helps to terminate low-performing runs early during hyperparameter tuning experiments, thereby saving time and resources.

### Results

#### Hyperdrive Run In Progress
![Hyperdrive Run Details In Progress](./screenshots/Hyperdrive%20Run%20Details%20Start.png)

#### Hyperdrive Run Complete
![Hyperdrive Run Details In Progress](./screenshots/Hyperdrive%20Run%20Details%20Complete.png)

#### Hyperdrive Results
![Hyperdrive Results](./screenshots/Hyperdrive%20Sweep.png)

#### Hyperdrive Best Run
![Hyperdrive Results](./screenshots/Hyperdrive%20Best%20Run.png)

### Register the best model
![Hyperdrive Best Model](./screenshots/Hyperdrive%20Best%20Model%20Registered.png)

## Model Deployment

Best Model from AutoML was deployed as it had the best accuracy.
#### Register the best model
![Best AutoML Model](./screenshots/AutoML%20Best%20Model%20Registered.png)

#### Deploy the best model
![Deployed Model](./screenshots/Model%20Endpoint.png)

### Webservice call
![Webservice Response](./screenshots/Webservice%20Call.png)

## Screen Recording
- Screen Recording   [Final Project Screencast](https://github.com/raman340/uda_final_project/blob/master/starter_file/Final%20Project.zip).

## Standout Suggestions


## Conclusion
This project holds potential in developing a predictive model for heart disease detection using the provided dataset. However, it is important to note some limitations and potential drawbacks before implementing this model in a real-world healthcare setting.

#### Pros of using this model in a real-world healthcare setting:

- Early identification of patients at risk of heart disease could lead to early intervention and prevention of heart disease.
- Automated detection of heart disease could lead to more efficient use of healthcare resources and improved patient outcomes.
- Machine learning models can analyze large amounts of data quickly, providing healthcare professionals with valuable insights into patient risk factors.

#### Cons of using this model in a real-world healthcare setting:

- The model is based on retrospective data, which may not accurately reflect the current population or demographic changes.
- The model's accuracy may be affected by differences in data collection across different hospitals and healthcare systems.
- The model's performance may degrade over time as patient populations and risk factors change.
- There may be ethical and legal considerations related to the use of machine learning models in healthcare decision-making.

In conclusion, our predictive model has shown promise in identifying patients at risk of heart disease. However, it is important to consider its limitations and potential drawbacks before implementing it in a real-world healthcare setting. Continuous validation and monitoring will be necessary to ensure its continued accuracy and usefulness.
