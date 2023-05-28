import pandas as pd
import scipy.stats
from fitter import Fitter
import numpy as np
import random
import scipy.stats as st

from empirical_distribution import EmpiricalDistribution
from skmultiflow.drift_detection import PageHinkley
from driftwatch import GreedyKS, IksReservoir, ReservoirSampling, LallDDSketch

import time
import dtaidistance.dtw
import collections as cl

import tqdm
import multiprocessing as mproc
import config

approx_methods = ['GreedyKS', 'Reservoir Sampling', 'IKS + RS', 'Page Hinkley']#, 'Lall + DDSketch']

results_drifts_file_names = ['detect_drifts_m_1_d_22_29_1683155601.pkl',
              'detect_drifts_m_1_d_29_32_1683155708.pkl',
              'detect_drifts_m_2_d_1_8_1683160209.pkl',
              'detect_drifts_m_2_d_8_15_1683160165.pkl',
              'detect_drifts_m_2_d_15_22_1683160111.pkl',
              'detect_drifts_m_2_d_22_29_1683160222.pkl',
              'detect_drifts_m_2_d_29_32_1683160325.pkl',
              'detect_drifts_m_3_d_1_8_1683218975.pkl',
              'detect_drifts_m_3_d_8_15_1683217859.pkl',
              'detect_drifts_m_3_d_15_22_1683219938.pkl',
              'detect_drifts_m_3_d_22_29_1683220334.pkl',
              'detect_drifts_m_3_d_29_32_1683217095.pkl',
              'detect_drifts_m_4_d_1_8_1683231359.pkl',
              'detect_drifts_m_4_d_8_15_1683230749.pkl',
              'detect_drifts_m_4_d_15_22_1683231360.pkl',
              'detect_drifts_m_4_d_22_29_1683232413.pkl',
              'detect_drifts_m_4_d_29_32_1683227988.pkl',
              'detect_drifts_m_5_d_1_8_1683243265.pkl',
              'detect_drifts_m_5_d_8_15_1683243191.pkl',
              'detect_drifts_m_5_d_15_22_1683242412.pkl',
              'detect_drifts_m_5_d_22_29_1683241935.pkl',
              'detect_drifts_m_5_d_29_32_1683236435.pkl',
              'detect_drifts_m_6_d_1_8_1683248575.pkl',
              'detect_drifts_m_6_d_8_15_1683343442.pkl',
              'detect_drifts_m_6_d_15_22_1683250960.pkl',
              'detect_drifts_m_6_d_22_29_1683252021.pkl',
              'detect_drifts_m_6_d_29_32_1683247203.pkl',
              'detect_drifts_m_7_d_1_8_1683294265.pkl',
              'detect_drifts_m_7_d_8_15_1683353629.pkl',
              'detect_drifts_m_7_d_15_22_1683295052.pkl',
              'detect_drifts_m_7_d_22_29_1683296056.pkl',
              'detect_drifts_m_7_d_29_32_1683289908.pkl',
              'detect_drifts_m_8_d_1_8_1683498475.pkl',
              'detect_drifts_m_8_d_8_15_1683497494.pkl',
              'detect_drifts_m_8_d_15_22_1683397061.pkl',
              'detect_drifts_m_8_d_22_29_1683396458.pkl',
              'detect_drifts_m_8_d_29_32_1683392343.pkl',
              'detect_drifts_m_9_d_1_8_1683405769.pkl',
              'detect_drifts_m_9_d_8_15_1683407329.pkl',
              'detect_drifts_m_9_d_15_22_1683406480.pkl',
              'detect_drifts_m_9_d_22_29_1683405600.pkl',
              'detect_drifts_m_9_d_29_32_1683399852.pkl',
              'detect_drifts_m_10_d_1_8_1683421974.pkl',
              'detect_drifts_m_10_d_8_15_1683419549.pkl',
              'detect_drifts_m_10_d_15_22_1683417274.pkl',
              'detect_drifts_m_10_d_22_29_1683418276.pkl',
              'detect_drifts_m_10_d_29_32_1683412878.pkl',
              'detect_drifts_m_11_d_1_8_1683429215.pkl',
              'detect_drifts_m_11_d_8_15_1683430757.pkl',
              'detect_drifts_m_11_d_15_22_1683432483.pkl',
              'detect_drifts_m_11_d_22_29_1683431999.pkl',
              'detect_drifts_m_11_d_29_32_1683424728.pkl',
              'detect_drifts_m_12_d_1_8_1683441243.pkl',
              'detect_drifts_m_12_d_8_15_1683441904.pkl',
              'detect_drifts_m_12_d_15_22_1683443181.pkl',
              'detect_drifts_m_12_d_22_29_1683441395.pkl',
              'detect_drifts_m_12_d_29_32_1683437793.pkl',
              ]


def ph_builder(ref_distrib, num_bins):
    return PageHinkley(min_instances=0)

def rs_builder(ref_distrib, num_bins):
    return ReservoirSampling(num_bins, ref_distrib)

def gks_builder(ref_distrib, num_bins):
    return GreedyKS(ref_distrib, num_bins, exact_prob=True)

# def dds_builder(ref_distrib, num_bins, stream):
#     return LallDDSketch(compute_ddsketch_error(stream, num_bins), ref_distrib)

def iks_builder(ref_distrib, num_bins):
    return IksReservoir(num_bins, ref_distrib)

method_factory = {
    'Page Hinkley': ph_builder,
    'Reservoir Sampling': rs_builder,
    'GreedyKS': gks_builder,
    # 'Lall + DDSketch': dds_builder,
    'IKS + RS': iks_builder,
}

def load_data(days_range, month, hours_range):
    months_with_30 = [4,6,9,11]

    stream = []
    days = range(*days_range)
    
    month_2_digits = '{month:02}'.format(month=month)
    path = '/media/thalis/arquivos linux/TCC/COVID19_Tweets_Dataset_2020/Summary_Sentiment/2020_' + month_2_digits + '/'

    for day in days:
        if month == 1 and day < 22:
            continue
        if month == 2 and day > 29:
            continue
        if day > 31:
            continue
        if month in months_with_30 and day > 30:
            continue

        for hour in range(*hours_range):
            file_name = path + f'2020_{month_2_digits}_' + '{day:02}'.format(day=day) \
                        + '_{hour:02}'.format(hour=hour) + '_Summary_Sentiment.pkl'
            try:
                df = pd.read_pickle(file_name)
                stream.append(df[config.file_column])
            except FileNotFoundError as ex:
                log_file = open('../logs.txt', 'a')
                log_file.write('File not found!: ' + file_name + '\n')
                log_file.close()
            

    return stream


def fit_data(data, type, predefined_dists=False):
    if predefined_dists:
        f = Fitter(data, distributions=['gumbel_r', 'laplace', 'logistic', 'norm', 'uniform'])
        # f.distributions = f.distributions[:2]
    else:
        f = Fitter(data)
    f.fit()

    if type == 'summary':
        return f.summary(method='ks_pvalue', plot=False, clf=False, Nbest=110)
    if type == 'get_best':
        return f.get_best()

def get_dist_obj(dist, param):
    return getattr(scipy.stats, dist)(**param)

def run(args):
    num_bins, tweets_per_file, dist_type, predefined_dists = args
    instances_methods = {}
    index = 0
    methods_drifts = {
        "Page Hinkley": [],
        "GreedyKS" : [],
        "Reservoir Sampling" : [],
        "IKS + RS": [],
        "mini-batch": []
    }
    methods_times = {
        "Page Hinkley": 0,
        "GreedyKS" : 0,
        "Reservoir Sampling" : 0,
        "IKS + RS" : 0
    }

    full_batch = []
    dist = None
    
    for tweets_hour in tqdm.tqdm(tweets_per_file):
        if len(tweets_hour) == 0:
            continue
        
        full_batch = np.concatenate((full_batch, tweets_hour))

        if dist == None or st.ks_1samp(full_batch, dist.cdf).pvalue < 0.01:
            if len(tweets_hour) > 3:
                if dist_type == 'fit':
                    best_fitted = fit_data(tweets_hour, 'get_best', predefined_dists=predefined_dists)
                    dist = get_dist_obj(list(best_fitted.keys())[0], list(best_fitted.values())[0])
                elif dist_type == 'empirical':
                    dist = EmpiricalDistribution(tweets_hour)
                else:
                    print('Unknow distribution type!')
                    return -1
                full_batch = tweets_hour
                methods_drifts['mini-batch'].append(index)
            else:
                dist = None
        
        for element in tweets_hour:
            index += 1
            for m in approx_methods:
                if m in instances_methods:
                    start = time.time()
                    instances_methods[m].add_element(element)
                    if instances_methods[m].detected_change():
                        methods_drifts[m].append(index)
                        del instances_methods[m]
                    
                    methods_times[m] = methods_times[m] + time.time() - start
        
        if len(tweets_hour) > 3:
            for m in approx_methods:
                if m not in instances_methods:
                    instances_methods[m] = method_factory[m](dist, num_bins)
                    for element in tweets_hour:
                        instances_methods[m].add_element(element)

    print('Number of tweets processed: ', index)
    return methods_drifts, methods_times
    # return {k:dtaidistance.dtw.distance(methods_drifts.get(k, []), methods_drifts['mini-batch']) for k in approx_methods if k != 'mini-batch'}, methods_times

def eval_call_center(args):
    ts_smp, num_bins = args
    
    instances_methods = {}
    resp = cl.defaultdict(list)
    
    minibatch_expon = None
    full_batch = []
    time = 0
    
    for i in ts_smp.groupby([ts_smp.dt.year, ts_smp.dt.month, ts_smp.dt.day, ts_smp.dt.hour]):
        latest_hour_batch = (i[1][1:].values - i[1][:-1].values).astype(float)/10**9
        latest_hour_var = len(set(latest_hour_batch))
        latest_hour_expon = None
        
        full_batch = np.concatenate((full_batch, latest_hour_batch))

        if minibatch_expon == None or st.ks_1samp(full_batch, minibatch_expon.cdf).pvalue < 0.01:
            if latest_hour_var > 3:
                latest_hour_expon = minibatch_expon = st.expon(*st.expon.fit(latest_hour_batch))
                full_batch = latest_hour_batch
                resp['mini-batch'].append(time)
            else:
                minibatch_expon = None

        for element in latest_hour_batch:
            time += 1
            for m in approx_methods:
                if m in instances_methods:
                    instances_methods[m].add_element(element)

                    if instances_methods[m].detected_change():
                        resp[m].append(time)
                        del instances_methods[m]

        if latest_hour_var > 3:
            for m in approx_methods:
                if m not in instances_methods:
                    latest_hour_expon = latest_hour_expon or st.expon(*st.expon.fit(latest_hour_batch))
                    
                    instances_methods[m] = method_factory[m](latest_hour_expon, num_bins, latest_hour_batch)

                    for element in latest_hour_batch:
                        instances_methods[m].add_element(element)

    return {k:dtaidistance.dtw.distance(resp.get(k, []), resp['mini-batch']) for k in approx_methods if k != 'mini-batch'}

def get_results_call_center(num_bins, samples, nproc=None):
    args_gen = ((sample, num_bins) for sample in samples)
    
    results = mproc.Pool(processes=nproc).imap(eval_call_center, args_gen)
    results = tqdm.tqdm(results, total=len(samples))
    return pd.DataFrame(results)