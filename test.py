from io import StringIO
import boto3
import pandas as pd
from aws_constants import AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY


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


belarus_weather_df = get_data(AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
print(belarus_weather_df.head(15))
