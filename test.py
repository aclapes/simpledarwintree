import numpy as np
import random
from operator import itemgetter

def point_distribution(n,m,random_seed=None):
    """
    Generate n m-dimensional points
    :param n: the number of points
    :param m: the number of dimensions for each point
    :return:
    """
    random.seed(random_seed)

    X = np.zeros((n,m), dtype=np.float32)
    for i in xrange(n):
        x = [random.random() for j in xrange(m)]
        X[i,:] = x / np.sum(x)

    return X

def subset_sum(numbers, target, partial=[]):
    s = sum(partial)

    # check if the partial sum is equals to target
    if s == target:
        print "sum(%s)=%s" % (partial, target)
    if s >= target:
        return  # if we reach the number why bother to continue

    for i in range(len(numbers)):
        n = numbers[i]
        remaining = numbers[i+1:]
        subset_sum(remaining, target, partial + [n])


if __name__ == "__main__":
    subset_sum(linspace(0))