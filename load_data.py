import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define the column names for the Adult dataset
COLUMNS = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
           'occupation', 'relationship', 'race', 'sex', 'capital-gain',
           'capital-loss', 'hours-per-week', 'native-country', 'income']

def load_dataset():

    # Load the data, drop unnecessary columns and missing values
    dataset_train = pd.read_csv('./adult.data', names=COLUMNS, na_values=[' ?']).drop(['fnlwgt'], axis=1).dropna()
    dataset_test = pd.read_csv('./adult.test', names=COLUMNS, na_values=[' ?']).drop(['fnlwgt'], axis=1).dropna()

    # Preprocess the training data
    features_train = pd.get_dummies(dataset_train.drop(['income'], axis=1))
    labels_train = pd.Categorical(dataset_train['income']).codes
    protected_attribute_train = pd.Categorical(dataset_train['sex']).codes

    # Preprocess the test data
    features_test = pd.get_dummies(dataset_test.drop(['income'], axis=1))
    labels_test = pd.Categorical(dataset_test['income']).codes
    protected_attribute_test = pd.Categorical(dataset_test['sex']).codes.astype('float32')

    # Align columns in test data with training data
    for column in features_train.columns:
        if column not in features_test.columns:
            features_test[column] = 0

    # Ensure same column order
    features_test = features_test[features_train.columns]

    # Standardize the data using only training data statistics
    scaler = StandardScaler()
    scaler.fit(features_train)

    features_train = scaler.transform(features_train)
    features_test = scaler.transform(features_test)

    return features_train, features_test, labels_train, labels_test, protected_attribute_train, protected_attribute_test