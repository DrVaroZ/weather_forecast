from io import StringIO
import boto3
import pandas as pd
from aws_constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly


def get_data(access_key_id: str, secret_access_key: str):
    s3_client = boto3.client(
        's3',
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key
    )

    response = s3_client.get_object(Bucket='weatherforecast-bucket',
                                    Key='GlobalWeatherRepository.csv')

    object_content = response['Body'].read().decode('utf-8')

    weather_df = pd.read_csv(StringIO(object_content))

    return weather_df.loc[weather_df.country == 'Belarus']


def transform_data(data):
    forecast_data = data.rename(columns={"last_updated": "ds",
                                         "temperature_celsius": "y"})
    forecast_data['ds'] = pd.to_datetime(forecast_data['ds']).dt.date
    forecast_data['ds'] = pd.to_datetime(forecast_data['ds'])
    forecast_data = forecast_data[['ds', 'y']]
    forecast_data.reset_index(inplace=True, drop=True)
    print(forecast_data.dtypes)
    print(forecast_data)
    return forecast_data


def make_forecast(forecast_data):
    model = Prophet()
    model.fit(forecast_data)
    forecasts = model.make_future_dataframe(periods=30)
    predictions = model.predict(forecasts)
    plot_plotly(model, predictions)
    return predictions


belarus_weather_df = get_data(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
# print(belarus_weather_df.head(15))
transformed_data = transform_data(belarus_weather_df)
predictions_forecast = make_forecast(transformed_data)
print(predictions_forecast)
