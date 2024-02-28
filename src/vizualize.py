import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

trainData = pd.read_csv('../dataset/cases_2021_train.csv')
testData = pd.read_csv('../dataset/cases_2021_test.csv')
location = pd.read_csv('../dataset/location_2021.csv')
fullSet = pd.concat([trainData,testData], ignore_index=True)
fullSet = fullSet.groupby("country")
# 11 cols

def parsePercentages(group):
    counts = {
        "age" : 0,
        "sex" :0,
        "province":0,
        "country":0,
        "latitude":0,
        "longitude":0,
        "date_confirmation":0,
        "additional_information":0,
        "source":0,
        "chronic_disease_binary":0,
        "outcome_group":0
    }
    for column in group.columns:
        counts[column] = group[column].count()
    percentages = {field: (count /  len(group)) * 100 for field, count in counts.items()}
    return counts, percentages

results = {}

for country, group in fullSet:
    counts, percentages = parsePercentages(group)
    results[country] = {
        "DataSize": len(group),
        "counts":counts,
        "percentages":percentages
    }

print(results)

