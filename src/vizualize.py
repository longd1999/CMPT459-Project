import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# Data Setup
trainData = pd.read_csv('../dataset/cases_2021_train.csv')
testData = pd.read_csv('../dataset/cases_2021_test.csv')
location = pd.read_csv('../dataset/location_2021.csv')
#https://github.com/samayo/country-json/blob/master/src/country-by-continent.json
countryMap = pd.read_json('../dataset/countryMap.json')

# Globals
fullSet = pd.concat([trainData,testData], ignore_index=True)
fullSet = pd.merge(fullSet, countryMap, on='country', how='left')
countrySet = fullSet.groupby("country")
fullSet = fullSet.groupby("continent")
results = {}
# This is a pre-config for the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# HELPERS

# Helper function, this is what is counting and calculating data that is currently present
def parsePercentages(group):
    counts = {}
    for column in group.columns:
        counts[column] = group[column].count()
    percentages = {field: (count /  len(group)) * 100 for field, count in counts.items()}
    total_percent = sum(percentages.values()) / 11
    return counts, percentages, total_percent

# Helper function to create heat maps, if you want to change something about the heat maps do it here
def createHeatMap(world, column, title, filename):
    fig, ax = plt.subplots()
    world.plot(column=column, cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
    ax.axis('off')
    ax.set_title(title, fontdict={'fontsize': '18', 'fontweight' : '3'})
    plt.savefig(filename)
    plt.close(fig) 

# Continent Logic
for country, group in fullSet:
    counts, percentages, total_percent = parsePercentages(group)
    results[country] = {
        "DataSize": len(group), # total data size per country
        "counts":counts, # counts data that exists vs what does not exist
        "percentages":percentages, # calculates percentage of data that is avaialable for each category
        "overall data": total_percent # calculates an overall "percent" of data that is present
    }
# Convert to data frame to work with data properly, need to use Transpose to set country as col. 
barResults = pd.DataFrame(results).T
# For bar plot of continents
plt.figure(figsize=(10, 8))
sns.barplot(x=barResults['overall data'], y=barResults.index)
plt.xlabel('Overall Data (%)')
plt.ylabel('Continent')
plt.title('Overall Data Percentage by Country')
plt.savefig('results.svg')

# Hetmap Logic

# Get various Stats
for country, group in countrySet:
    counts, percentages, total_percent = parsePercentages(group)
    country_index = world.index[world['name'] == country].tolist()
    if country_index:
        world.at[country_index[0], 'overallPercent'] = total_percent
        world.at[country_index[0], 'agePercent'] = percentages['age']
        world.at[country_index[0], 'sexPercent'] = percentages['sex']
        world.at[country_index[0], 'outcome'] = percentages['outcome']
# Lots of countries missing, need to fill them with something otherwise they just dont appear on maps
world['overallPercent'].fillna(0, inplace=True)
world['agePercent'].fillna(0, inplace=True)
world['sexPercent'].fillna(0, inplace=True)
world['outcome'].fillna(0, inplace=True)
# Heatmaps
createHeatMap(world, 'overallPercent', 'Overall Data Reported', 'Overall.svg')
createHeatMap(world, 'agePercent', 'Countries Reporting Age', 'Countries.svg')
createHeatMap(world, 'sexPercent', 'Countries Reporting Sex', 'Sex.svg')
createHeatMap(world, 'outcome', 'Countries Reporting Outcomes', 'Outcome.svg')
