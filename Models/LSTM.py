import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import plot_tree
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import keras_tuner as kt
import tensorflow as tf

#Data reading and cleaning
print("Reading and cleaning data")
data=pd.read_csv('BruteForce.csv')

data[" Label"]=data[" Label"].apply(lambda x:0 if x=='BENIGN' else 1)
print("Encoded the label Class for Binary Classification")
data.drop_duplicates(inplace=True)

# Dropping duplicates and NaN values and handling inf values
data.dropna(inplace=True)
data.drop_duplicates(inplace=True)
data=data.replace([np.inf, -np.inf], np.finfo(np.float32).max)
# print("Dropped duplicates and NaN values and handled inf values")
irrelevant_columns =[ 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', ' act_data_pkt_fwd', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']
data.drop(columns=irrelevant_columns, inplace=True)
print("Dropped irrelevant columns")


#Handling Data Imbalance
print("Handling Data Imbalance")
# rus=RandomUnderSampler(sampling_strategy={0:5000,1:5000},random_state=42)
rus=RandomUnderSampler(sampling_strategy=1,random_state=42)
print("Data shape after cleaning:",data.shape)
print("Class counts after cleaning and Under sampling to handel class Imbalance:") 
print(data[" Label"].value_counts())

#x and y split and train test split
print("Splitting data into x and y")
scaler = StandardScaler()
x=data.drop(columns=[" Label"])
y=data[" Label"]
x = scaler.fit_transform(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
x_train, y_train = rus.fit_resample(x_train, y_train)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
print("Tranning data shape:",x_train.shape)
print(y_train.value_counts())

def build_lstm_model(hp):
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    lstm_units = hp.Int('lstm_units', min_value=32, max_value=256, step=32)
    dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=128)
    dropout_rate = hp.Float('dropout_rate', 0.3, 0.7, step=0.1)

    lstm = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x.shape[1], 1)),
        tf.keras.layers.LSTM(lstm_units, return_sequences=True),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')  # For binary classification
    ])

    lstm.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return lstm



tuner = kt.RandomSearch(
    build_lstm_model,
    objective='val_accuracy',
    max_trials=2,
    executions_per_trial=1,
    directory='LSTN_Trials',
    project_name='LSTN_Trials/WebAttacks_LSTM',
)
tuner.search(x_train, y_train, epochs=10, validation_data=(x_test, y_test), batch_size=32)
best_model = tuner.get_best_models(num_models=1)[0]
best_model.summary()
loss,acc=best_model.evaluate(x_test, y_test)
print("Final Test Accuracy:", acc)
print("Final Test Loss:", loss)
pred=best_model.predict(x_test)
confusion_matrix = tf.math.confusion_matrix(y_test, np.round(pred), num_classes=2)  
cm = confusion_matrix.numpy()
print("Confusion Matrix:")
print(cm)
print("f1 score:",f1_score(y_test, np.round(pred)))
print("recall score:",recall_score(y_test, np.round(pred)) )
print("precision score:",accuracy_score(y_test, np.round(pred)))