import numpy as np
from scipy.stats import t, norm

def ci_half_length(n,alpha,sample_std):
    delta = t.ppf(1-alpha/2, n-1, loc=0, scale=1) * sample_std/np.sqrt(n) 
    return delta

def approximate_num_reps_relative1(random_sample,gamma,alpha,n0):
    num_samples = len(random_sample)
    sample_std = np.std(random_sample)
    sample_mean = np.mean(random_sample)
    n = n0
    if num_samples < n0:
        print('too few samples, need at least 5')
    else:
        n = np.ceil( ((sample_std*t.ppf(1-alpha/2, num_samples-1, loc=0, scale=1)) / (gamma*np.abs(sample_mean)) )**2 )
    return n

def approximate_num_reps_relative2(random_sample,gamma,alpha,n0):
    num_samples = len(random_sample)
    sample_std = np.std(random_sample)
    sample_mean = np.mean(random_sample)
    n = n0
    if num_samples < n0:
        print('too few samples, need at least 5')
    else:
        while  ci_half_length(n,alpha,sample_std) / np.abs(sample_mean) > gamma/(1-gamma):
            n += 1
    return n

def approximate_num_reps_absolute(random_sample,beta,alpha,n0):
    num_samples = len(random_sample)
    sample_std = np.std(random_sample)
    sample_mean = np.mean(random_sample)
    n = n0
    if num_samples < n0:
        print('too few samples, need at least 5')
    else:
        while  ci_half_length(n,alpha,sample_std)  > beta:
            n += 1
    return n
    
def sequental_num_reps_relative(random_sample,gamma,alpha,n0): #now I give the big samples as it is cheap, can also call the simulation ON DEMAND
    num_samples = len(random_sample)
    n = n0
    if num_samples < n0:
        print('too few samples, need at least 5')
    else:
        sample_std = np.std(random_sample[0:n])
        sample_mean = np.mean(random_sample[0:n])
        while  ci_half_length(n,alpha,sample_std) / np.abs(sample_mean) > gamma/(1-gamma):
            n += 1
            sample_std = np.std(random_sample[0:n])
            sample_mean = np.mean(random_sample[0:n])
    return n
