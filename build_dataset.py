import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

def build_dataset(test_size=0.2, random_state=42):
    # load the Adult dataset from 'adult.csv' and define the columns
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
        'hours-per-week', 'native-country', 'income'
    ]
    data = pd.read_csv('adult.csv', names=column_names, sep=',', engine='python') # the separator ',' is used, and `engine='python'` is specified to handle potential whitespace issues
    
    # replace missing values (indicated by '?') with NaN and drop rows with missing values
    # this cleans the data by removing incomplete records
    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)
    
    # separate the features (X) from the target labels (y)
    # `income` column is the target variable, while all other columns are features
    X = data.drop('income', axis=1)
    y = data['income']
    
    # convert the `income` column to binary labels:
    # '>50K' is encoded as 1 (indicating high income) and '<=50K' as 0 (indicating low income)
    y = y.apply(lambda x: 1 if x == '>50K' else 0).values.reshape(-1, 1)
    
    # identify categorical columns (non-numeric) in the features for label encoding
    categorical_cols = X.select_dtypes(include=['object']).columns

    # separate categorical and numerical columns to process them separately
    X_categorical = X[categorical_cols]
    X_numerical = X.drop(columns=categorical_cols)
    
    # apply label encoding to convert categorical variables to numerical form
    # each categorical feature is converted into integer labels to be processed by the model
    X_categorical = X_categorical.apply(LabelEncoder().fit_transform)
    
    # combine the numerical and encoded categorical columns back into a single DataFrame
    # this forms the final set of processed features
    X_processed = pd.concat([X_numerical, X_categorical], axis=1)
    X_values = X_processed.values.astype(float)
    
    # standardize the features to have zero mean and unit variance
    # this scaling improves model performance by normalizing feature ranges
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_values)
    
    # split the data into training and testing sets
    # `test_size` specifies the proportion of data to be used for testing
    # `random_state` ensures reproducibility of the split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
