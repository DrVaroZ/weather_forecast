import pandas as pd


user_data_df = pd.read_csv("D:/Python projects/rootkey.csv")

AWS_ACCESS_KEY_ID = user_data_df.iloc[0]['Access key ID']
AWS_SECRET_ACCESS_KEY = user_data_df.iloc[0]['Secret access key']
