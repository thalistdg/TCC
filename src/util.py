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

approx_methods = ['Page Hinkley','GreedyKS', 'Reservoir Sampling', 'IKS + RS']#, 'Lall + DDSketch']

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

def load_data(file_column, days_range, months_range, hours_range, sampling_p=1.0):
    stream = []
    days = range(*days_range)
    months = range(*months_range)
    for month in months:
        month_2_digits = '{month:02}'.format(month=month)
        path = '../COVID19_Tweets_Dataset_2020/Summary_Sentiment/2020_' + month_2_digits + '/'

        for day in days:
            if month == 1 and day < 22:
                continue
            if month == 2 and day > 29:
                continue
            if day > 31:
                continue

            for hour in range(*hours_range):
                file_name = path + f'2020_{month_2_digits}_' + '{day:02}'.format(day=day) \
                            + '_{hour:02}'.format(hour=hour) + '_Summary_Sentiment.csv'
                df = pd.read_csv(file_name,
                                 names=['Tweet_ID','Sentiment_Label','Logits_Neutral','Logits_Positive','Logits_Negative'],
                                 header=0,
                                 skiprows=(lambda x:random.random() > sampling_p)
                                )
                stream.append(df[file_column])

    return (pd.concat(stream, ignore_index=True), stream) if len(stream) > 0 else (stream, stream)


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