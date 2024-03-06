import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

# get and merge data
trainData = pd.read_csv('../dataset/cases_2021_train.csv')
testData = pd.read_csv('../dataset/cases_2021_test.csv')
location = pd.read_csv('../dataset/location_2021.csv')
#https://github.com/samayo/country-json/blob/master/src/country-by-continent.json
countryMap = pd.read_json('../dataset/countryMap.json')
fullSet = pd.concat([trainData,testData], ignore_index=True)
fullSet = pd.merge(fullSet, countryMap, on='country', how='left')
fullSet = fullSet.groupby("continent")

# 11 cols

# helper function, this is what is counting and calculating data that is currently present
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
        "DataSize": len(group), # total data size per country
        "counts":counts, # counts data that exists vs what does not exist
        "percentages":percentages, # calculates percentage of data that is avaialable for each category
        "overall data": total_percent # calculates an overall "percent" of data that is present
    }

# Convert to data frame to work with data properly, need to use Transpose to set country as col. 
results = pd.DataFrame(results).T

# For bar plot of continents

plt.figure(figsize=(10, 8))
sns.barplot(x=results['overall data'], y=results.index)
plt.xlabel('Overall Data (%)')
plt.ylabel('Continent')
plt.title('Overall Data Percentage by Country')
plt.savefig('results.svg')



world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# Generate random values for each country
np.random.seed(0)  # Set seed for reproducibility
world['random_value'] = np.random.rand(len(world))

# Plotting
fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.plot(column='random_value', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8')
ax.axis('off')

fig, ax = plt.subplots(1, 1, figsize=(15, 10))
world.plot(column='random_value', cmap='YlOrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
ax.axis('off')

# Add a title
ax.set_title('World Heat Map with Random Values for Each Country', fontdict={'fontsize': '18', 'fontweight' : '3'})


# Show the plot
plt.savefig('temp.svg')

# Load population data (assuming it's in a DataFrame called population_data)
# You might need to adjust the column names accordingly
population_data = results['DataSize']

# Merge population data with world map data
world = world.merge(population_data, how='left', left_on='name', right_index=True)



# TODO: switch current bar chart to use seaborn https://seaborn.pydata.org/examples/part_whole_bars.html
# TODO: Group countries into "continents" which is not a current param in data
# TODO: Create heat map, the data that is currently in "results" should hopefully make that pretty easy