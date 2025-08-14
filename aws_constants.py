import os
import pandas as pd

AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')

if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    try:
        rootkey_csv_path = os.getenv('AWS_ROOTKEY_CSV', 'rootkey.csv')
        user_data_df = pd.read_csv(rootkey_csv_path)
        AWS_ACCESS_KEY_ID = user_data_df.iloc[0]['Access key ID']
        AWS_SECRET_ACCESS_KEY = user_data_df.iloc[0]['Secret access key']
    except Exception:
        AWS_ACCESS_KEY_ID = None
        AWS_SECRET_ACCESS_KEY = None
