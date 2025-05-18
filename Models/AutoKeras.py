import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
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
import keras_tuner as kt
import tensorflow as tf
import autokeras as ak

#Data reading and cleaning
print("Reading and cleaning data")
data=pd.read_csv('Infilteration.csv')

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
scaler = StandardScaler()
x=data.drop(columns=[" Label"])
x=scaler.fit_transform(x)
y=data[" Label"]
print("Label counts after cleaning and Under sampling to handle class Imbalance:")
print(y.value_counts())
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train, y_train = rus.fit_resample(x_train, y_train)
print("y_train label counts after cleaning and Under sampling to handle class Imbalance:")
print(y_train.value_counts())
y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

#Auto keras model building

input_node=ak.Input()
output_node=ak.ClassificationHead()(input_node)
model=ak.AutoModel(inputs=input_node,outputs=output_node,max_trials=5,directory='autokeras_model',project_name='Infilteration',objective='val_accuracy')
model.fit(x_train,y_train,epochs=20,validation_data=(x_test,y_test))
best_model=model.export_model()
best_model.summary()
loss,acc=best_model.evaluate(x_test, y_test)
print("Final Test Accuracy:", acc)
print("Final Test Loss:", loss)
pred=best_model.predict(x_test)
print("F1 Score:", f1_score(y_test, np.round(pred)))
print("Precision Score:", precision_score(y_test, np.round(pred)))
print("Recall Score:", recall_score(y_test, np.round(pred)))
confusion_matrix = tf.math.confusion_matrix(y_test, np.round(pred), num_classes=2)
cm = confusion_matrix.numpy()
print("Confusion Matrix:")
print(cm)