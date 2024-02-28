import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

trainData = pd.read_csv('../dataset/cases_2021_train.csv')
testData = pd.read_csv('../dataset/cases_2021_test.csv')
location = pd.read_csv('../dataset/location_2021.csv')
fullSet = pd.concat([trainData,testData], ignore_index=True)
fullSet = fullSet.groupby("country")

for country, group in fullSet:
    print("Country:", country)
    print(group)