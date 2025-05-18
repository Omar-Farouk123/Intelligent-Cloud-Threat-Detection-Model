# import pandas as pd
# import numpy as np
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# from sklearn.calibration import LabelEncoder
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from imblearn.over_sampling import SMOTE


# bot=pd.read_csv("Bot.csv")
# BruteForce=pd.read_csv("BruteForce.csv")
# bot_brute=pd.merge(bot, BruteForce, how='outer')
# DDoS=pd.read_csv("DDoS.csv")
# bot_brute_ddos=pd.merge(bot_brute, DDoS, how='outer')
# Dos=pd.read_csv("DoS.csv")
# bot_brute_ddos_dos=pd.merge(bot_brute_ddos, Dos, how='outer')
# Infilteration=pd.read_csv("Infiltration.csv")
# bot_brute_ddos_dos_infiltration=pd.merge(bot_brute_ddos_dos, Infilteration, how='outer')
# PortScan=pd.read_csv("PortScan.csv")
# bot_brute_ddos_dos_infiltration_portscan=pd.merge(bot_brute_ddos_dos_infiltration, PortScan, how='outer')
# WebAttacks=pd.read_csv("WebAttacks.csv")
# finalMerge=pd.merge(bot_brute_ddos_dos_infiltration_portscan, WebAttacks, how='outer')

# finalMerge.drop_duplicates(inplace=True)
# finalMerge.dropna(inplace=True)

# print(finalMerge[Label].value_counts())
# print(finalMerge.shape)
