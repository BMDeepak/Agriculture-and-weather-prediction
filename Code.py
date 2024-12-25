# Import Libraries
import requests
import pandas as pd
from dagster import asset
import psycopg2
import json
from pymongo import MongoClient
import seaborn as sns
import matplotlib.pyplot as plt
from meteostat import Point, Daily
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# Load Data Function
@asset
def loaddata():
    dublin = Point(53.3498, -6.2603)
    start = datetime(2022, 1, 1)
    end = datetime(2022, 12, 31)

    # Fetch Weather Data
    weather_data = Daily(dublin, start, end).fetch()
    weather_data.reset_index(inplace=True)
    weather_data.rename(columns={'time': 'datetime'}, inplace=True)
    print("Weather Data Loaded:")
    print(weather_data.head())

    # Load Agriculture Data
    agr_data = pd.read_csv('C:\\Users\\shiva\\Documents\\Dataanalytics\\Data analytics project\\FAOSTAT_data-2022.csv')
    print("Agriculture Data Loaded:")
    print(agr_data.head())

    return {
        'weather_data': weather_data,
        'agr_data': agr_data
    }


# Save Weather Data to PostgreSQL
@asset
def saveweathertopostgres(loaddata: dict):
    weather_data = loaddata["weather_data"]

    weather_json = weather_data.to_json(orient="records")
    weather_records = json.loads(weather_json)

    # Database Connection
    conn = psycopg2.connect(
        dbname="weather_db",
        user="postgres",
        password="root",
        host="localhost",
        port="5432"
    )
    cursor = conn.cursor()

    # Create Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id SERIAL PRIMARY KEY,
            data JSONB
        );
    """)

    # Insert Data
    for record in weather_records:
        cursor.execute(
            "INSERT INTO weather_data (data) VALUES (%s);",
            [json.dumps(record)]
        )

    conn.commit()
    cursor.close()
    conn.close()
    print("Weather data successfully saved to PostgreSQL!")


# Save Agriculture Data to MongoDB
@asset
def saveagriculturetomongo(loaddata: dict):
    agr_data = loaddata.get("agr_data")

    if not isinstance(agr_data, pd.DataFrame):
        raise ValueError("Agriculture data must be a pandas DataFrame.")

    try:
        client = MongoClient("mongodb://localhost:27017/")
        db = client['agriculture_db']
        collection = db['agriculture_data']

        agr_records = agr_data.to_dict(orient='records')
        collection.insert_many(agr_records)

        print("Agriculture data successfully saved to MongoDB!")
    except Exception as e:
        print(f"Error saving agriculture data: {e}")
    finally:
        client.close()


# Data Cleaning
@asset
def datacleaning(loaddata: dict):
    weather_data = loaddata["weather_data"]
    agr_data = loaddata["agr_data"]

    # Weather Data Cleaning
    weather_data['datetime'] = pd.to_datetime(weather_data['datetime'], errors='coerce')
    weather_data['temp_range'] = weather_data['tmax'] - weather_data['tmin']
    weather_data = weather_data.drop(['snow', 'tsun'], axis=1, errors='ignore')
    weather_data = weather_data.dropna(subset=['datetime', 'tmax', 'tmin', 'prcp', 'wpgt'])

    # Agriculture Data Cleaning
    agr_data['value_per_unit'] = agr_data['Value'] / agr_data['Unit'].map({'ha': 1, 'kg/ha': 1000, 't': 1})
    agr_data_filtered = agr_data[agr_data['Element'].isin(['Yield', 'Production'])].dropna(subset=['Item'])
    agr_data_encoded = pd.get_dummies(agr_data_filtered, columns=['Item'], drop_first=True)

    # Merge Datasets
    weather_data['year'] = weather_data['datetime'].dt.year
    weather_aggregated = weather_data.groupby('year').agg({
        'tavg': 'mean',
        'tmin': 'mean',
        'tmax': 'mean',
        'prcp': 'sum'
    }).reset_index()

    merged_data = pd.merge(agr_data_encoded, weather_aggregated, left_on='Year', right_on='year', how='inner')
    merged_data = merged_data.drop(columns=['year'])

    return merged_data
