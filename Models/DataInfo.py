import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE



data=pd.read_csv('WebAttacks.csv')
data[" Label"]=data[" Label"].apply(lambda x:0 if x=='BENIGN' else 1)
x=data.drop(columns=[" Label"])
y=data[" Label"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
rus=RandomUnderSampler(sampling_strategy=1,random_state=42)
x_train, y_train = rus.fit_resample(x_train, y_train)
print(y.shape)
print(y.value_counts())
print(y_train.shape)
print(y_train.value_counts())
print(y_test.shape)
print(y_test.value_counts())
