import csv
import pandas as pd
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests as req
# lib to make string url safe
import urllib.parse

countries_dict = {}

invalid_countries = {
    'Taiwan*': 'Taiwan',
    'Korea, South': 'South Korea',
    'US': 'United States',
    'Congo (Brazzaville)': 'Democratic Republic of the Congo',
    'Congo (Kinshasa)': 'Democratic Republic of the Congo',
    'MS Zaandam': 'Unknown',
    'West Bank and Gaza': 'Palestine',
    'Summer Olympics 2020': 'Unknown',
    'Holy See': 'Vatican City',
    'Burma': 'Myanmar',
    'Sao Tome and Principe': 'São Tomé and Príncipe',
    'Cabo Verde': 'Cape Verde',
    'Diamond Princess': 'Unknown',
    'Cote d\'Ivoire': 'Ivory Coast',
    '' : 'Unknown',
}

def update_countries_population_dict(countries):
    if countries is not None:
        for country in countries:
            if country not in countries_dict:
                if country in invalid_countries:
                    # print (f'Country {country} is found invalid, using {invalid_countries[country]} instead for api call')
                    country = invalid_countries[country]
                if country == 'Taiwan':
                    countries_dict[country] = {
                        'population': 23576775,
                        'continent': 'Asia',
                        'count' : 1,
                        'data_availability': '0%'
                    }
                elif country == 'Unknown':
                    continue
                else:
                    country_info = get_population_continent(country)
                    # if country info is of type int
                    if country_info is not None and isinstance(country_info, int):
                        print(f'Country {country} not found')
                    else:
                        countries_dict[country] = {
                            'population': country_info[0],
                            'continent': country_info[1] if country_info[1] else 'Unknown',
                            'count' : 1,
                            'data_availability': '0%'
                        }
                


# function to call api with country name and get the population
def get_population_continent(country):
    country = country.strip()
    country = country.lower()
    # make country url safe
    country = urllib.parse.quote(country)

    #call get api on restcountries.com/v3.1/name/{country}?fullText=true
    response = req.get(f'https://restcountries.com/v3.1/name/{country}?fullText=true')
    if response.status_code == 200:
        data = response.json()
        if len(data) > 0:
            return data[0]['population'], data[0]['continents'][0]
        else:
            return 0
    else:
        return 0


def open_file(file_path):
    try:
        with open(file_path, newline='', encoding='utf-8') as csvfile:
            reader = csv.reader(csvfile)
            data = [row for row in reader]
            return data
    except FileNotFoundError:
        print('File not found')
        return None
    except Exception as e:
        print(e)
        return None


def combine_test_train():
    trainFile = open_file('/Users/manvirheer/sfu/project459spring2024/project_desc_files/csvs/cases_2021_train.csv');
    testFile = open_file('/Users/manvirheer/sfu/project459spring2024/project_desc_files/csvs/cases_2021_test.csv')
    # test trainFile and testFile
    if trainFile is None or testFile is None:
        print('Error reading files')
    else:
        trainFile = pd.DataFrame(trainFile[1:], columns=trainFile[0])
        testFile = pd.DataFrame(testFile[1:], columns=testFile[0])
        combined = pd.concat([trainFile, testFile])
        # update the column labels
        combined.rename(columns={'latitude': 'case_lats', 'longitude': 'case_longs'}, inplace=True)
        # add the combined_key column
        combined['Combined_Key'] = combined['province'] + ', ' + combined['country']
        unique_countries = combined['country'].unique()
        update_countries_population_dict(unique_countries)
        return combined

def count_countries(dataFrame):
    for index, row in dataFrame.iterrows():
        # if row have country column

        if 'country' in row:
            if row['country'] in invalid_countries:
                row['country'] = invalid_countries[row['country']]
            if row['country'] in countries_dict:
                countries_dict[row['country']]['count'] += 1
            elif row['country'] == 'Unknown':
                continue
            else:
                countries_dict[row['country']] = {
                    'population': 0,
                    'count': 1,
                    'data_availability': '0%'
                }
        if 'Country_Region' in row:
            if row['Country_Region'] in invalid_countries:
                row['Country_Region'] = invalid_countries[row['Country_Region']]
            if row['Country_Region'] in countries_dict:
                countries_dict[row['Country_Region']]['count'] += 1
            elif row['Country_Region'] == 'Unknown':
                continue
            else:
                countries_dict[row['Country_Region']] = {
                    'population': 0,
                    'count': 1,
                    'data_availability': '0%'
                }
    return countries_dict

def calculate_missing_incident_rate(dataFrame):
    count = 0
    for index, row in dataFrame.iterrows():
        if row['Incident_Rate'] == '':
            count += 1
            if row['Confirmed'] != '' and row['Confirmed'] != '0':
                dataFrame.at[index, 'Incident_Rate'] = (int(row['Confirmed']) / int(countries_dict[row]['population'])) * 100000
            else:
                dataFrame.at[index, 'Incident_Rate'] = 0

    print(f'Calculated {count} missing incident rates')
    return dataFrame   

def process_combined_data(dataFrame):
    dataFrame.drop_duplicates(subset=None, keep="first", inplace=False)
    for index, row in dataFrame.iterrows():
        # if the combined_key doesn't have province info before comma, then remove comma and trim
        if row['Combined_Key'][0] == ',':
            dataFrame.at[index, 'Combined_Key'] = row['Combined_Key'][1:].strip()
        # if the age is a range with a dash, then split the range and take the average
        if row['age'] and '-' in row['age']:
            age_range = row['age'].split('-')
            # if the dash has only one number before and after, then take the number as the age
            if len(age_range) == 2 and age_range[0].isdigit() and age_range[1].isdigit():
                dataFrame.at[index, 'age'] = (int(age_range[0]) + int(age_range[1])) / 2
            else:
                dataFrame.at[index, 'age'] = int(age_range[0])

        if row['province'] == 'Taiwan':
            dataFrame.at[index, 'country'] = 'Taiwan'
            dataFrame.at[index, 'Combined_Key'] = 'Taiwan'
            dataFrame.at[index, 'province'] = ''

    count_countries(dataFrame)
    dataFrame.drop_duplicates(subset=None, keep="first", inplace=False)
    dataFrame.to_csv('/Users/manvirheer/sfu/project459spring2024/tasks/task1/added_reference_files/cases_2021_combined.csv', index=False)
    return dataFrame        
        

def read_locationsCSV():
    locations = open_file('/Users/manvirheer/sfu/project459spring2024/project_desc_files/csvs/location_2021.csv')
    if locations is None:
        print('Error reading files')
    else:
        locations = pd.DataFrame(locations[1:], columns=locations[0])
        #update str to float 
        locations['Expected_Mortality_Rate'] = locations['Deaths'].astype('float') / locations['Confirmed'].astype('float')
        update_countries_population_dict(locations['Country_Region'].unique())
        count_countries(locations)
        locations.to_csv('/Users/manvirheer/sfu/project459spring2024/tasks/task1/added_reference_files/processed_location_2021.csv', index=False)
        return locations
   
def add_countries_data_availability(countries_dict):
    for country in countries_dict:
        data_availability_percentage = 0

        if countries_dict[country]['count'] > 0 and countries_dict[country]['population'] > 0:
            # only keep 6 decimal places after the decimal point
            data_availability_percentage = float(f'{(countries_dict[country]["count"] / countries_dict[country]["population"]):.7f}')
        
        countries_dict[country]['data_availability'] = data_availability_percentage
        
# create bar graph for each continent based on the number of cases with the x-axis being the continent and the y-axis being the number of cases and more cases should be darker in color
def create_bar_graph(countries_dict):
    continent_cases = {}
    for country in countries_dict:
        continent = countries_dict[country]['continent']
        if continent in continent_cases:
            continent_cases[continent] += countries_dict[country]['count']
        else:
            continent_cases[continent] = countries_dict[country]['count']
    continent_cases = {k: v for k, v in sorted(continent_cases.items(), key=lambda item: item[1], reverse=True)}
    fig = px.bar(x=list(continent_cases.keys()), y=list(continent_cases.values()), title='Cases per Continent', labels={'x': 'Continent', 'y': 'Number of Cases'})
    fig.show()


def create_heatmap(data):
    # Convert the dictionary data into a Pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index').reset_index().rename(columns={'index': 'country'})
    #sort on data_availability and print top 10
    df = df.sort_values(by='data_availability', ascending=False)
    print(df.head(10))
    min_value = df['data_availability'].min()
    max_value = df['data_availability'].max()
    # remove vatican city
    df = df[df['country'] != 'Vatican City']
    df['log_data_availability'] = np.log10(df['data_availability'] + 1)  # Adding 1 to avoid log(0)

    fig = px.choropleth(df, locations="country",
                        locationmode="country names",
                        color="data_availability",
                        color_continuous_scale=px.colors.sequential.Plasma,
                        range_color=((min_value ), (0.00002 )),
                        title="Data Availability by Country (Log Scale)")
    fig.show()
    

combined_DF = combine_test_train()
processed_combined_data = process_combined_data(combined_DF)
read_locationsCSV()
add_countries_data_availability(countries_dict)
create_heatmap(countries_dict)
# print those countries which are having 0 population
count_cases = 0


print(f'Total cases: {count_cases}')
print(countries_dict)
create_bar_graph(countries_dict)
