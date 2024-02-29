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
    counts = {}
    for column in group.columns:
        counts[column] = group[column].count()
    percentages = {field: (count /  len(group)) * 100 for field, count in counts.items()}
    total_percent = sum(percentages.values()) / 11
    return counts, percentages, total_percent

results = {}

for country, group in fullSet:
    counts, percentages, total_percent = parsePercentages(group)
    results[country] = {
        "DataSize": len(group),
        "counts":counts,
        "percentages":percentages,
        "overall data": total_percent
    }

results = pd.DataFrame(results).T

plt.figure(figsize=(10, 8))
plt.barh(results.index, results['overall data'])
plt.xlabel('Overall Data (%)')
plt.title('Overall Data Percentage by Country')
plt.savefig('results.svg')
