import random
from bisect import bisect_left
from scipy import stats
from scipy.stats import norm
import numpy as np

class EmpiricalDistribution():
    def __init__(self, data):
        self.data = sorted(data)

    def cdf(self, x):
        if(type(x) != list and type(x) != np.ndarray):
            return bisect_left(self.data, x)/len(self.data)
        
        return np.array([bisect_left(self.data, v)/len(self.data) for v in x])
        
    def rvs(self, size):
        return np.around(random.sample(self.data, size), 8)
