import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import joblib 

#Data reading and cleaning
print("Reading and cleaning data")
data=pd.read_csv('WebAttacks.csv')

data[" Label"]=data[" Label"].apply(lambda x:0 if x=='BENIGN' else 1)
print("Encoded the label Class for Binary Classification")
data.drop_duplicates(inplace=True)

# Dropping duplicates and NaN values and handling inf values
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data=data.replace([np.inf, -np.inf], np.finfo(np.float32).max)
print("Dropped duplicates and NaN values and handled inf values")


#Handling Data Imbalance
print("Handling Data Imbalance")
rus=RandomUnderSampler(sampling_strategy=1,random_state=42)
print("Data shape after cleaning:",data.shape)
print("Class counts after cleaning and Under sampling to handel class Imbalance:") 
print(data[" Label"].value_counts())

#x and y split and train test split
print("Splitting data into x and y")
x=data.drop(columns=[" Label"])
y=data[" Label"]
#under sampling to handle class imbalance
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train, y_train = rus.fit_resample(x_train, y_train)
print("Tranning data shape:",x_train.shape)
print(y_train.value_counts())

#Decision Tree Training and Evaluation
print("Training and Evaluating Decision Tree Classifier")
dt=DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42)
dt.fit(x_train,y_train)

tree_train_predictions = dt.predict(x_train)
tree_test_predictions = dt.predict(x_test)
print("Decision Tree Classifier:")
print("Training Accuracy:", accuracy_score(y_train, tree_train_predictions))
print("Test Accuracy:", accuracy_score(y_test, tree_test_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, tree_test_predictions))


#dropping low importance features
print("Dropping low importance features")
importances = dt.feature_importances_
indices = np.argsort(importances)[::-1]
print("Feature ranking:")

low_importance_features = [x.columns[i] for i in range(len(importances)) if importances[i] < 0.001]
print("Low importance features:")
print(low_importance_features)
data.drop(columns=low_importance_features, inplace=True)
print("Dropped low importance features")
print("Data shape after dropping low importance features:",data.shape)

#Decision Tree Training and Evaluation after dropping low importance features
print("Training and Evaluating Decision Tree Classifier after dropping low importance features")
x=data.drop(columns=[" Label"])
y=data[" Label"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
dt=DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42,class_weight='balanced')
dt.fit(x_train,y_train)

tree_train_predictions = dt.predict(x_train)
tree_test_predictions = dt.predict(x_test)
print("Decision Tree Classifier:")
print("Training Accuracy:", accuracy_score(y_train, tree_train_predictions))
print("Test Accuracy:", accuracy_score(y_test, tree_test_predictions))
print("F1 Score:", f1_score(y_test, tree_test_predictions))
print("Precision Score:", precision_score(y_test, tree_test_predictions))
print("Recall Score:", recall_score(y_test, tree_test_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, tree_test_predictions))



# Random Forest Classifier
print("Training and Evaluating Random Forest Classifier")
rf=RandomForestClassifier(n_estimators=200,min_samples_leaf=5,min_samples_split=20,max_depth=4,class_weight="balanced", random_state=42)
rf.fit(x_train,y_train)
rf_train_predictions = rf.predict(x_train)
rf_test_predictions = rf.predict(x_test)
print("Random Forest Classifier:")
print("Training Accuracy:", accuracy_score(y_train, rf_train_predictions))
print("Test Accuracy:", accuracy_score(y_test, rf_test_predictions))
print("F1 Score:", f1_score(y_test, rf_test_predictions))
print("Precision Score:", precision_score(y_test, rf_test_predictions))
print("Recall Score:", recall_score(y_test, rf_test_predictions))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_test_predictions))

