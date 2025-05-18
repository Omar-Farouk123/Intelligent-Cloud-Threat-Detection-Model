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
irrelevant_columns =[' Flow Duration', ' Total Fwd Packets', ' Total Backward Packets', 'Total Length of Fwd Packets', ' Total Length of Bwd Packets', ' Fwd Packet Length Max', ' Fwd Packet Length Min', ' Fwd Packet Length Mean', ' Fwd Packet Length Std', ' Bwd Packet Length Min', ' Bwd Packet Length Mean', ' Flow Packets/s', ' Flow IAT Mean', ' Flow IAT Std', ' Flow IAT Max', ' Flow IAT Min', 'Fwd IAT Total', ' Fwd IAT Mean', ' Fwd IAT Std', ' Fwd IAT Max', 'Bwd IAT Total', 
' Bwd IAT Mean', ' Bwd IAT Max', 'Fwd PSH Flags', ' Bwd PSH Flags', ' Fwd URG Flags', ' Bwd URG Flags', ' Fwd Header Length', ' Bwd Header Length', 'Fwd Packets/s', ' Bwd Packets/s', ' Min Packet Length', ' Max Packet Length', ' Packet Length Mean', ' Packet Length Std', ' Packet Length Variance', 'FIN Flag Count', ' SYN Flag Count', ' RST Flag Count', ' PSH Flag Count', ' ACK Flag Count', ' URG Flag Count', ' CWE Flag Count', ' ECE Flag Count', ' Down/Up Ratio', ' Average Packet Size', ' Avg Bwd Segment Size', ' Fwd Header Length.1', 'Fwd Avg Bytes/Bulk', ' Fwd Avg Packets/Bulk', ' Fwd Avg Bulk Rate', ' Bwd Avg Bytes/Bulk', ' Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets', ' Subflow Fwd Bytes', ' Subflow Bwd Packets', ' Subflow Bwd Bytes', ' act_data_pkt_fwd', 'Active Mean', ' Active Std', ' Active Max', ' Active Min', 'Idle Mean', ' Idle Std', ' Idle Max', ' Idle Min']
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
x_train = x_train[..., np.newaxis]
x_test=x_test[...,np.newaxis] 
print("Tranning data shape:",x_train.shape)
print(y_train.value_counts())

def build_cnn_model(hp):
    optimizer = hp.Choice('optimizer', ['adam', 'rmsprop', 'sgd'])
    dense_units = hp.Int('dense_units', min_value=128, max_value=512, step=128)
    dropout_rate = hp.Float('dropout_rate', 0.3, 0.7, step=0.1)

    cnn = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(x.shape[1], 1)),
        tf.keras.layers.Conv1D(
            filters=hp.Int('filters', min_value=64, max_value=256, step=64),
            kernel_size=hp.Int('kernel_size', min_value=3, max_value=5, step=1),
            activation='relu',
            padding='same'  # Ensure output shape matches input shape
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(
            filters=hp.Int('filters_2', min_value=64, max_value=256, step=64),
            kernel_size=hp.Int('kernel_size_2', min_value=3, max_value=5, step=1),
            activation='relu',
            padding='same'  # Same padding to ensure dimension compatibility
        ),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(dense_units, activation='relu'),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Correct for binary classification
    ])

    cnn.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',  # Correct for binary
        metrics=['accuracy']
    )

    return cnn



tuner = kt.RandomSearch(
    build_cnn_model,
    objective='val_accuracy',
    max_trials=5,
    executions_per_trial=1,
    directory='cnn_trials',
    project_name='WebAttacks_CNN'
)
tuner.search(x_train, y_train, epochs=20, validation_data=(x_test, y_test), batch_size=32)
best_model = tuner.get_best_models(num_models=1)[0]
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