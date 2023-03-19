import pandas as pd
import scipy.stats
from fitter import Fitter

def load_data(file_column, days_range, months_range, hours_range):
    stream = []
    days = range(*days_range)
    months = range(*months_range)
    for month in months:
        month_2_digits = '{month:02}'.format(month=month)
        path = '../COVID19_Tweets_Dataset_2020/Summary_Sentiment/2020_' + month_2_digits + '/'

        for day in days:
            if month == 2 and day > 29:
                continue
            if month == 1 and day < 22:
                continue

            for hour in range(*hours_range):
                file_name = path + f'2020_{month_2_digits}_' + '{day:02}'.format(day=day) + '_{hour:02}'.format(hour=hour) + '_Summary_Sentiment.csv'
                stream.append(pd.read_csv(file_name)[file_column])

    return (pd.concat(stream, ignore_index=True), stream) if len(stream) > 0 else (stream, stream)


def fit_data(data, type, test=False):
    f = Fitter(data)
    if test:
        f.distributions = f.distributions[:2]
    f.fit()

    if type == 'summary':
        return f.summary(method='ks_pvalue', plot=False, clf=False, Nbest=110)
    if type == 'get_best':
        return f.get_best()

def get_dist_obj(dist, param):
    return getattr(scipy.stats, dist)(**param)