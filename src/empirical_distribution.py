import random
from bisect import bisect_left
from scipy import stats


class EmpiricalDistribution():
    def __init__(self, data):
        self.data = sorted(data)

    def cdf(self, x):
        return bisect_left(self.data, x)/len(self.data)

    def rvs(self, size):
        return random.sample(self.data, size)



dist = EmpiricalDistribution([10,20,9,28,5,2,8,37,34,30])
cdf = dist.cdf(38)
print(cdf)
print(dist.rvs(1))