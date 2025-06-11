import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score  
from sklearn.calibration import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.tree import DecisionTreeClassifier

# Load the merged merged_dfset
merged_df = pd.read_csv("DTRF_Merged.csv")
merged_df.drop_duplicates(inplace=True)

# Dropping duplicates and NaN values and handling inf values
merged_df.dropna(inplace=True)
merged_df.drop_duplicates(inplace=True)
merged_df=merged_df.replace([np.inf, -np.inf], np.finfo(np.float32).max)
print("Dropped duplicates and NaN values and handled inf values")


#Handling merged_df Imbalance
print("Handling merged_df Imbalance")
print("merged_df shape after cleaning:",merged_df.shape)
print("Class counts after cleaning and Under sampling to handel class Imbalance:") 
print(merged_df["Label"].value_counts())

#x and y split and train test split
print("Splitting merged_df into x and y")
x=merged_df.drop(columns=["Label"])
y=merged_df["Label"]
print("total rows in y:", len(y))
#under sampling to handle class imbalance
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
print("Train and Test split Counts:",y_train.shape, y_test.shape)
count= y_train.value_counts().min()
samplingstrategy = {
    'Normal': count,
    'DDoS': count,
    'DoS': count,
    'PortScan': count,
    'BruteForce': count
}
rus=RandomUnderSampler(sampling_strategy=samplingstrategy,random_state=42)
x_train, y_train = rus.fit_resample(x_train, y_train)
print("Tranning merged_df shape after balancing classes:",x_train.shape)
print("Class counts after balancing classes:")
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
print("F1 Score:", f1_score(y_test, tree_test_predictions,average='macro'))
print("Precision Score:", precision_score(y_test, tree_test_predictions,average='macro'))
print("Recall Score:", recall_score(y_test, tree_test_predictions,average='macro'))
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
merged_df.drop(columns=low_importance_features, inplace=True)
print("Dropped low importance features")
print("merged_df shape after dropping low importance features:",merged_df.shape)

#Decision Tree Training and Evaluation after dropping low importance features
print("Training and Evaluating Decision Tree Classifier after dropping low importance features")
x=merged_df.drop(columns=["Label"])
y=merged_df["Label"]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state=42)
dt=DecisionTreeClassifier(max_depth=10, min_samples_split=20, random_state=42,class_weight='balanced')
dt.fit(x_train,y_train)

tree_train_predictions = dt.predict(x_train)
tree_test_predictions = dt.predict(x_test)
print("Decision Tree Classifier after dropping low important columns:")
print("Training Accuracy:", accuracy_score(y_train, tree_train_predictions))
print("Test Accuracy:", accuracy_score(y_test, tree_test_predictions))
print("F1 Score:", f1_score(y_test, tree_test_predictions,average='macro'))
print("Precision Score:", precision_score(y_test, tree_test_predictions,average='macro'))
print("Recall Score:", recall_score(y_test, tree_test_predictions,average='macro'))
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
print("F1 Score:", f1_score(y_test, rf_test_predictions,average='macro'))
print("Precision Score:", precision_score(y_test, rf_test_predictions,average='macro'))
print("Recall Score:", recall_score(y_test, rf_test_predictions,average='macro'))
print("Confusion Matrix:")
print(confusion_matrix(y_test, rf_test_predictions))

