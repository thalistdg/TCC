# import requests
from urllib.error import HTTPError
import tqdm
import pandas as pd
import random

# input month number to download
month = 12
month = '{month:02}'.format(month=month)

# write days for this month.
first_day_in_month = 1
last_day_in_month = 31

sampling_p = 0.1

for day in tqdm.tqdm(range(first_day_in_month, last_day_in_month+1)):
    day_2_d = '{day:02}'.format(day=day)
    for hour in range(0,24):
        hour_2_d = '{hour_2:02}'.format(hour_2=hour)
        url = 'https://raw.githubusercontent.com/lopezbec/COVID19_Tweets_Dataset_2020/' \
            + f'master/Summary_Sentiment/2020_{month}/2020_{month}_{day_2_d}_{hour_2_d}_Summary_Sentiment.csv'

        # req = requests.get(url)

        try:
            req = pd.read_csv(url,
                            names=['Tweet_ID','Sentiment_Label','Logits_Neutral','Logits_Positive','Logits_Negative'],
                            header=0,
                            skiprows=(lambda x:random.random() > sampling_p),
                            usecols=['Logits_Negative'],
                            )
        except HTTPError as ex:
            if ex.code == 404:
                file = open('../logs.txt', 'a')
                file.write('Error 404 in url: ' + url + '\n')
                file.close()
                continue
            else:
                raise ex

        path = f'/media/thalis/arquivos linux/TCC/COVID19_Tweets_Dataset_2020/Summary_Sentiment/2020_{month}/'
        file_name = f'2020_{month}_{day_2_d}_{hour_2_d}_Summary_Sentiment.pkl'
        
        pd.to_pickle(req, path + file_name)
