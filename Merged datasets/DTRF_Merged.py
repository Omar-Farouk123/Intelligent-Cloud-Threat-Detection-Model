import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.calibration import LabelEncoder
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#Bot dataset
bot=pd.read_csv("Bot.csv")
print("Bot dataset label counts:")
bot[" Label"]=bot[" Label"].apply(lambda x:'Normal' if x=='BENIGN' else 'Bot')
print(bot[' Label'].value_counts())

#BruteForce dataset
BruteForce=pd.read_csv("BruteForce.csv")
print("BruteForce dataset label counts:")
BruteForce[" Label"]=BruteForce[" Label"].apply(lambda x:'Normal' if x=='BENIGN' else 'BruteForce')
print(BruteForce[' Label'].value_counts())

#DDoS dataset
DDoS=pd.read_csv("DDoS.csv")
print("DDoS dataset label counts:")
DDoS[" Label"]=DDoS[" Label"].apply(lambda x:'Normal' if x=='BENIGN' else 'DDoS')
print(DDoS[' Label'].value_counts())

#DoS dataset
Dos=pd.read_csv("DoS.csv")
print("DoS dataset label counts:")
Dos[" Label"]=Dos[" Label"].apply(lambda x:'Normal' if x=='BENIGN' else 'DoS')
print(Dos[' Label'].value_counts())

#Infilteration dataset
Infilteration=pd.read_csv("Infilteration.csv")
print("Infiltration dataset label counts:")
Infilteration[" Label"]=Infilteration[" Label"].apply(lambda x:'Normal' if x=='BENIGN' else 'Infilteration')
print(Infilteration[' Label'].value_counts())

#PortScan dataset
PortScan=pd.read_csv("PortScan.csv")
print("PortScan dataset label counts:")
PortScan[" Label"]=PortScan[" Label"].apply(lambda x:'Normal' if x=='BENIGN' else 'PortScan')
print(PortScan[' Label'].value_counts())

#WebAttacks dataset
WebAttacks=pd.read_csv("WebAttacks.csv")
print("WebAttacks dataset label counts:")
WebAttacks[" Label"]=WebAttacks[" Label"].apply(lambda x:'Normal' if x=='BENIGN' else 'WebAttack')
print(WebAttacks[" Label"].value_counts())


merged_df = pd.concat([BruteForce, DDoS, Dos, PortScan], ignore_index=True)

# Rename ' Label' to 'Label' (remove leading space)
merged_df.rename(columns={" Label": "Label"}, inplace=True)

# Show class distribution
print("Final merged dataset label counts:")
print(merged_df["Label"].value_counts())
print(merged_df.shape)


#saving the merged dataset
# print("Saving the merged dataset to DTRF_Merged.csv")
# merged_df.to_csv("DTRF_Merged.csv", index=False)