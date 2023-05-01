import requests
import tqdm

# input month number to download
month = 8
month = '{month:02}'.format(month=month)

# write last day for this month. e.g. 31 or 30 or 29.
last_day_in_month = 31

for day in tqdm.tqdm(range(1, last_day_in_month+1)):
    day_2_d = '{day:02}'.format(day=day)
    for hour in range(24):
        hour_2_d = '{hour_2:02}'.format(hour_2=hour)
        url = 'https://raw.githubusercontent.com/lopezbec/COVID19_Tweets_Dataset_2020/' \
            + f'master/Summary_Sentiment/2020_{month}/2020_{month}_{day_2_d}_{hour_2_d}_Summary_Sentiment.csv'

        req = requests.get(url)

        path = f'./COVID19_Tweets_Dataset_2020/Summary_Sentiment/2020_{month}/'
        file_name = f'2020_{month}_{day_2_d}_{hour_2_d}_Summary_Sentiment.csv'
        csv_file = open(path + file_name, 'wb')
        csv_file.write(req.content)
        csv_file.close()