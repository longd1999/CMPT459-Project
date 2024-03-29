import csv
import pandas as pd
import plotly.express as px
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import requests as req
from datetime import datetime
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

map_outcome = {
    'Hospitalized': 'Hospitalized',
    'Recovered': 'Discharged',
    'Deceased': 'Deceased',
    'recovered': 'Discharged',
    'died': 'Deceased',
    'Under treatment': 'Hospitalized',
    'Receiving Treatment': 'Hospitalized',
    'Alive': 'Hospitalized',
    'discharge': 'Discharged',
    'stable': 'Hospitalized',
    'stable condition': 'Hospitalized',
    'discharged': 'Discharged',
    'death': 'Deceased',
    'Stable': 'Hospitalized',
    'Dead': 'Deceased',
    'Died': 'Deceased',
    'Death': 'Deceased',
    'Discharged from hospital': 'Discharged',
    'released from quarantine': 'Discharged',
    'Discharged': 'Discharged',
    'recovering at home 03.03.2020': 'Discharged',
    'critical condition': 'Hospitalized'
}

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
                        'data_availability': '0%',
                        'Lat': 23.6978,
                        'Long_': 120.9605

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
                            'population': country_info[0] if country_info[0] else 0,
                            'continent': country_info[1] if country_info[1] else 'Unknown',
                            'count' : 1,
                            'data_availability': '0%',
                            'Lat': country_info[2] if country_info[2] else 0,
                            'Long_': country_info[3] if country_info[3] else 0
                        }
                

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
            return data[0]['population'], data[0]['continents'][0], data[0]['latlng'][0], data[0]['latlng'][1]
        else:
            return 0
    else:
        return 0


def combine_test_train(train_file_path, test_file_path):
    trainFile = open_file(train_file_path)
    testFile = open_file(test_file_path)

    # test trainFile and testFile
    if trainFile is None or testFile is None:
        print('Error reading files')
        return None;

    trainFile = pd.DataFrame(trainFile[1:], columns=trainFile[0])
    #testFile = pd.DataFrame(testFile[1:], columns=testFile[0])

    # combine the two dataframes
    #combined = pd.concat([trainFile, testFile])
    combined = trainFile
    # update the column labels
    combined.rename(columns={'latitude': 'case_lats', 'longitude': 'case_longs'}, inplace=True)

    # add the combined_key column
    combined['Combined_Key'] = ''

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
                print(f'Country {row["country"]} not found in countries_dict')
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
    dataFrame = dataFrame.reset_index(drop=True)
    count = 0
    for index, row in dataFrame.iterrows():
        if row['Incident_Rate'] == '' and row['Country_Region'] != 'Unknown':
            count += 1
            if row['Confirmed'] != '' and row['Confirmed'] != '0':
                try:
                    confirmedCases = int(row['Confirmed'])
                    population = countries_dict[row['Country_Region']]['population']
                    dataFrame.at[index, 'Incident_Rate'] = (confirmedCases / population) * 100000
                except ZeroDivisionError:
                    dataFrame.at[index, 'Incident_Rate'] = 0
                except:
                    print('Error calculating incident rate for country: ' + row['Country_Region'] + ' with population: ' + str(population) + ' dic object is: ' + str(countries_dict[row['Country_Region']]));
            else:
                dataFrame.at[index, 'Incident_Rate'] = 0

    print(f'Calculated {count} missing incident rates')
    return dataFrame   


def process_combined_data(dataFrame, updatedCombinedFilePath):
    dataFrame = dataFrame.reset_index(drop=True)
    dataFrame.drop_duplicates(subset=None, keep="first", inplace=False)
    indices_to_drop = []
    for index, row in dataFrame.iterrows():
        # if the country is invalid, then update it to the correct country
        if row['country'] in invalid_countries:
            dataFrame.at[index, 'country'] = invalid_countries[row['country']]
        # if the combined_key doesn't have province info before comma, then remove comma and trim
        if row['Combined_Key'] == '':
            if row['province'] != '':
                dataFrame.at[index, 'Combined_Key'] = row['province'] + ', ' + row['country']
            else:
                dataFrame.at[index, 'Combined_Key'] = row['country']
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
            
        dataFrame.at[index, 'outcome_group'] = map_outcome[row['outcome_group']] if row['outcome_group'] in map_outcome else row['outcome_group']
        if type(row['outcome']) != str:
            print(f'Outcome is not a string: {row["outcome"]}')
            # print whole row
            print(row)
        else:
            # make outcome all lowercase and first letter uppercase
            dataFrame.at[index, 'outcome'] = row['outcome'].lower().capitalize().strip() if row['outcome'] else 'Unknown'
   
        # if age or sex is missing, then add to indices_to_drop
        if row['age'] == '' or row['sex'] == '':
            indices_to_drop.append(index)
            
        # if any column is numeric then round it to 4 decimal places
        for column in dataFrame.columns:
            if dataFrame[column].dtype == 'float64':
                dataFrame.at[index, column] = round(row[column], 4)

    dataFrame.drop(list(indices_to_drop), inplace=True, errors='ignore')
    dataFrame.reset_index(drop=True)
    unique_countries = dataFrame['country'].unique()
    update_countries_population_dict(unique_countries)
    count_countries(dataFrame)

    dataFrame.to_csv(updatedCombinedFilePath, index=False)
    return dataFrame        
        

def process_location_date(dataFrame, invalid_countries):
    dataFrame = dataFrame.reset_index(drop=True)
    # Drop duplicates and missing values upfront
    dataFrame = dataFrame.drop_duplicates(subset=None, keep="first", inplace=False)
    indices_to_drop = []
    newEntries = {}
    last_key = None
    for index, row in dataFrame.iterrows():
        country = row['Country_Region']
        province = row['Province_State']
        deaths = recovered = active = confirmed = 0

        if country in invalid_countries:
            corrected_country = invalid_countries[country]
            dataFrame.at[index, 'Country_Region'] = corrected_country
            if row['Province_State'] != '':
                dataFrame.at[index, 'Combined_Key'] = province + ', ' + corrected_country
            else:
                dataFrame.at[index, 'Combined_Key'] = corrected_country

        # Special case for Taiwan
        if province == 'Taiwan':
            dataFrame.at[index, 'Country_Region'] = 'Taiwan'
            dataFrame.at[index, 'Combined_Key'] = 'Taiwan'
            dataFrame.at[index, 'Province_State'] = ''

        if row['Lat'] == '' or row['Long_'] == '':
            if country in countries_dict:
                dataFrame.at[index, 'Lat'] = countries_dict[country]['Lat']
                dataFrame.at[index, 'Long_'] = countries_dict[country]['Long_']   
                
        for column in dataFrame.columns:
            if dataFrame[column].dtype == 'float64':
                dataFrame.at[index, column] = round(row[column], 4)
    
    dataFrame = dataFrame.dropna(subset=['Country_Region'])
    
    return dataFrame


def handle_location_data_duplicates(dataFrame):
    last_key = None
    newEntries = {}
    indices_to_drop = []
    for index, row in dataFrame.iterrows():
        current_key = row['Country_Region'] + '_' + row['Province_State']
        

        if current_key == last_key:
            indices_to_drop.extend([index, index - 1])
            if last_key in newEntries:
                time1 = datetime.strptime(row['Last_Update'], '%Y-%m-%d %H:%M:%S')
                time2 = datetime.strptime(newEntries[last_key]['Last_Update'], '%Y-%m-%d %H:%M:%S')
                newTime = time1 if time1 > time2 else time2
                newTime = newTime.strftime('%Y-%m-%d %H:%M:%S')

                newEntries[last_key]['Deaths'] += safe_convert_to_int(row['Deaths'])
                newEntries[last_key]['Recovered'] += safe_convert_to_int(row['Recovered'])
                newEntries[last_key]['Active'] += safe_convert_to_int(row['Active'])
                newEntries[last_key]['Confirmed'] += safe_convert_to_int(row['Confirmed'])
                newEntries[last_key]['Incident_Rate'] = newEntries[last_key]['Confirmed'] / countries_dict[row['Country_Region']]['population'] * 100000
                newEntries[last_key]['Case_Fatality_Ratio'] = newEntries[last_key]['Deaths'] / newEntries[last_key]['Confirmed'] * 100 if newEntries[last_key]['Confirmed'] > 0 else 0
                newEntries[last_key]['Expected_Mortality_Rate'] = newEntries[last_key]['Deaths'] / newEntries[last_key]['Confirmed'] if newEntries[last_key]['Confirmed'] > 0 else 0
                newEntries[last_key]['Last_Update'] = newTime
                indices_to_drop.append(index)
            else:
                time1 = datetime.strptime(row['Last_Update'], '%Y-%m-%d %H:%M:%S')
                time2 = datetime.strptime(dataFrame.at[index - 1, 'Last_Update'], '%Y-%m-%d %H:%M:%S')
                newTime = time1 if time1 > time2 else time2
                newTime = newTime.strftime('%Y-%m-%d %H:%M:%S')
                newConfirmed =  safe_convert_to_int(row['Confirmed']) + safe_convert_to_int(dataFrame.at[index - 1, 'Confirmed'])
                newDeaths = safe_convert_to_int(row['Deaths']) + safe_convert_to_int(dataFrame.at[index - 1, 'Deaths'])
                newRecovered = safe_convert_to_int(row['Recovered']) + safe_convert_to_int(dataFrame.at[index - 1, 'Recovered'])
                newActive = safe_convert_to_int(row['Active']) + safe_convert_to_int(dataFrame.at[index - 1, 'Active'])

                try:
                    newIncidentRate = newConfirmed / countries_dict[row['Country_Region']]['population'] * 100000
                except:
                    newIncidentRate = 0
                    
                try:
                    newEntries[last_key] = {
                        'Country_Region': row['Country_Region'],
                        'Province_State': row['Province_State'],
                        'Lat': row['Lat'],
                        'Long_': row['Long_'],
                        'Last_Update': newTime,
                        'Deaths': newDeaths,
                        'Recovered': newRecovered,
                        'Active': newActive,
                        'Combined_Key': row['Combined_Key'],
                        'Confirmed': newConfirmed,
                        'Incident_Rate': newIncidentRate,
                        'Case_Fatality_Ratio': newDeaths / newConfirmed * 100 if newConfirmed > 0 else 0,
                        'Expected_Mortality_Rate': newDeaths / newConfirmed if newConfirmed > 0 else 0
                    }
                except:
                    print('Error creating new entry')

        else:
            if last_key in newEntries and index - 1 not in indices_to_drop:
                indices_to_drop.append(index - 1)

        last_key = current_key
    # Remove identified duplicates
    dataFrame.drop(list(indices_to_drop), inplace=True, errors='ignore')
    
    dataFrame.reset_index(drop=True)

    newEntries = pd.DataFrame(list(newEntries.values()))
    dataFrame = pd.concat([dataFrame, newEntries], ignore_index=True)
    return dataFrame


def safe_convert_to_int(value):
    try:
        return int(value)
    except ValueError:
            if type(value) == str:
                print(' Unable to convert to int: ' + value)
                if '.' in value:
                    return int(float(value))
                if ',' in value:
                    if '.' in value:
                        return int(float(value.replace(',', '')))
            if pd.isnull(value):
                print('Value is null')
                return 0
            return 0


def read_locationsCSV(locationFilePath, updatedLocationsFilePath):
    locations = open_file(locationFilePath)
    if locations is None:
        print('Error reading files')
    else:
        locations = pd.DataFrame(locations[1:], columns=locations[0])

        # Assign 0 to missing values in the Deaths, Recovered, Active, and Confirmed columns
        locations.fillna({'Confirmed': 0, 'Deaths': 0, 'Active': 0, 'Recovered': 0}, inplace=True)
        #update str to float 
        locations['Expected_Mortality_Rate'] = locations['Deaths'].astype('float') / locations['Confirmed'].astype('float')
        update_countries_population_dict(locations['Country_Region'].unique())

        locations = process_location_date(locations, invalid_countries=invalid_countries)

        locations = calculate_missing_incident_rate(locations)
        locations = handle_location_data_duplicates(locations)
        locations.to_csv(updatedLocationsFilePath, index=False)
        return locations


def add_countries_data_availability(countries_dict):
    for country in countries_dict:
        data_availability_percentage = 0

        if countries_dict[country]['count'] > 0 and countries_dict[country]['population'] > 0:
            # only keep 6 decimal places after the decimal point
            data_availability_percentage = float(f'{(countries_dict[country]["count"] / countries_dict[country]["population"]):.7f}')
        
        countries_dict[country]['data_availability'] = data_availability_percentage
        

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

        
train_file_path = '/Users/manvirheer/sfu/cmpt459spring2024/CMPT459-Project/task2/project_desc_files/csvs/cases_2021_train.csv'
test_file_path = '/Users/manvirheer/sfu/cmpt459spring2024/CMPT459-Project/task2/project_desc_files/csvs/cases_2021_test.csv'
locationFilePath = '/Users/manvirheer/sfu/cmpt459spring2024/CMPT459-Project/task2/project_desc_files/csvs/location_2021.csv'
updatedLocationsFilePath = '/Users/manvirheer/sfu/cmpt459spring2024/CMPT459-Project/task2/added_reference_files/updated_location_2021.csv'
updatedCombinedFilePath = '/Users/manvirheer/sfu/cmpt459spring2024/CMPT459-Project/task2/added_reference_files/updated_combined_2021.csv'
combined_DF = combine_test_train(train_file_path=train_file_path, test_file_path=test_file_path)
processed_combined_data = process_combined_data(combined_DF, updatedCombinedFilePath)

processed_location_data = read_locationsCSV(locationFilePath, updatedLocationsFilePath=updatedLocationsFilePath)
# track the unmerged data
merged_df = pd.merge(processed_combined_data, processed_location_data, on='Combined_Key', how='left', suffixes=('', '_drop'))
merged_df = merged_df.loc[:, ~merged_df.columns.str.endswith('_drop')]
# drop column country_region
merged_df = merged_df.drop(columns='Country_Region')
merged_df = merged_df.drop(columns = 'Province_State')
Philippines_row = processed_location_data.loc[processed_location_data['Country_Region'] == 'Philippines']
track_row = []
indexToDrop = []
for index, row in merged_df.iterrows():
    
    if row['country'] == 'Philippines':
        row['Last_Update'] = Philippines_row['Last_Update'].values[0]
        row['Deaths'] = Philippines_row['Deaths'].values[0]
        row['Recovered'] = Philippines_row['Recovered'].values[0]
        row['Active'] = Philippines_row['Active'].values[0]
        row['Confirmed'] = Philippines_row['Confirmed'].values[0]
        row['Incident_Rate'] = Philippines_row['Incident_Rate'].values[0]
        row['Case_Fatality_Ratio'] = Philippines_row['Case_Fatality_Ratio'].values[0]
        row['Expected_Mortality_Rate'] = Philippines_row['Expected_Mortality_Rate'].values[0]
        row['Lat'] = Philippines_row['Lat'].values[0]
        row['Long_'] = Philippines_row['Long_'].values[0]
    if row['Lat'] == '' or pd.isna(row['Lat']):
       track_row.append({'country': row['country'], 'province': row['province']})
       indexToDrop.append(index)

merged_df = merged_df.drop(indexToDrop)
merged_df.reset_index(drop=True)

# drop duplicates from track_row
track_row = [dict(t) for t in {tuple(d.items()) for d in track_row}]
    
# print track_row
for row in track_row:
    print(row)
# drop one of the duplicate Combined_Key columns
merged_df.to_csv('/Users/manvirheer/sfu/cmpt459spring2024/CMPT459-Project/task2/added_reference_files/merged_data.csv', index=False)
add_countries_data_availability(countries_dict)

