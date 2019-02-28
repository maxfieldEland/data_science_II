# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 12:06:05 2019
Script to answer DS 2 Homework Question 2
@author: mgreen13
"""

# Libraries needed for assignment
import numpy as np
import matplotlib.pyplot as plt
import pymc3 as pm
# --------------------------------------------- Problem Number 1 ------------------------------------

# Define battles and losses
mk1_battles = 26751
mk1_deaths = 183

mk2_battles = 27079
mk2_deaths = 222
# Generate binary columns from stats
mk1_data = [0]*(mk1_battles-mk1_deaths) + [1]*mk1_deaths
mk2_data = [0]*(mk2_battles-mk2_deaths) + [1]*mk2_deaths

# initiate pymc3 model
with pm.Model() as model:
    
    mk1 = pm.Uniform('mk1',lower = 0, upper = 1)
    mk2 = pm.Uniform('mk2',lower = 0, upper = 1)

    obs1 = pm.Bernoulli('obs1', mk1, observed = mk1_data)
    obs2 = pm.Bernoulli('obs2',mk2, observed = mk2_data)
    # define diff as difference between mk1 and mk2
    diff = pm.Deterministic("diff", mk1 - mk2)
    
    # begin mcmc
    step = pm.Metropolis()
    trace = pm.sample(20000,step=step)
    burned_trace = trace[1000:]
    

mk1_samples = burned_trace["mk1"]
mk2_samples = burned_trace["mk2"]

plt.hist(mk1_samples,alpha = .8,label = 'mk1 samples')
plt.hist(mk2_samples,alpha = .8,label = 'mk2 samples')
plt.legend(loc = "upper right")
plt.title("Death Rate Distributions of Warriors from MK Hunters")
plt.show()

delta_samples = burned_trace["delta"]

#histogram of posteriors
pm.traceplot(trace)
pm.gelman_rubin(trace)


print("Probability Mk2 is deadlier than MK1 B: %.3f" % np.mean(delta_samples < 0))

# ------------------------------------------------------ Problem Number 2 --------------------------------------
# Load data into memory
    
count_data = np.loadtxt("data/txtdata.csv")
n_count_data = len(count_data)
plt.bar(np.arange(n_count_data), count_data, color="#348ABD")
plt.xlabel("Time (days)")
plt.ylabel("count of text-msgs received")
plt.title("Did the user's texting habits change over time?")
plt.xlim(0, n_count_data);

# Initiate pymc3 model building to build MCMC paramters lambda1,lambda2, and Tau.

with pm.Model() as model:
    alpha = 1.0/count_data.mean()  # Recall count_data is the
                                   # variable that holds our txt counts
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    
    tau = pm.DiscreteUniform("tau", lower=0, upper=n_count_data - 1)
    idx = np.arange(n_count_data) # Iday number ID
    # choose lambda_ as based on the  value sampled from the Tau random variable
    lambda_ = pm.math.switch(tau > idx, lambda_1, lambda_2)
    observation = pm.Poisson("obs", lambda_, observed=count_data)
    step = pm.Metropolis()
    trace = pm.sample(10000, tune=5000,step=step)
    
# save the sample values from lambda_1, lambda_2 and Tau
lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
tau_samples = trace['tau']
# generate traceplot from samples
pm.traceplot(trace)
# run gelman_rubin function to test for convergence
pm.gelman_rubin(trace)

# part 2.2

def switch_function(idx,l1,l2,f1,f2):
    """ 
        INPUT: 
            idx : time (t) in days
            l1  : random variable sample, lambda_1
            l2  : random variable sample, lambda_2
            f1  : random variable sample, phi_1
            f2  : random variable sample, phi_2
        OUTPUT: 
            Function outputs the logit equation transormed by lambda_1 and translated by lambda_2.
            This value is an estimate of the txt messages sent on a given day.
    """
    return(l1/(1+np.exp(idx*f1+f2))+l2)
    
# Open model block, generate parameters.
with pm.Model() as model:
    alpha = 1.0/count_data.mean()  
                                   
    lambda_1 = pm.Exponential("lambda_1", alpha)
    lambda_2 = pm.Exponential("lambda_2", alpha)
    
    phi_1 = pm.Normal('phi_1',0,3)
    phi_2 = pm.Normal('phi_2',0,3)
    # Impliment new lamda switch function in place of sampling from Tau.
    lambda_new = pm.Deterministic("lambda_new",switch_function(idx,lambda_1,lambda_2,phi_1,phi_2))

    idx = np.arange(n_count_data) 
    # the new partition, cut_off is the logit transformed lambda. 
    cut_off = switch_function(idx,lambda_1,lambda_2,phi_1,phi_2)
    lambda_ = pm.math.switch(cut_off > idx, lambda_1, lambda_2)
    # link data and prior
    observation = pm.Poisson("obs", lambda_, observed=count_data)
    # begin mcmc
    step = pm.Metropolis()
    trace = pm.sample(100000, tune=5000,step=step)
# save traces 
    
lambda_1_samples = trace['lambda_1']
lambda_2_samples = trace['lambda_2']
lambda_new_samples = trace['lambda_new']
phi_1_samples = trace['phi_1']
phi_2_samples = trace['phi_2']
# generate traceplot
pm.traceplot(trace)
# check convergence 
pm.gelman_rubin(trace)

# Create boolean mask to conduct weighted average of sampled lambda distribtuions
# Because lambda1/2 are the poisson parameter, this will return the expected value of 
# the message count of the day.

N = len(lambda_new_samples[:,0])
expected_texts_per_day = np.zeros(n_count_data)
for day in range(0, n_count_data):
    ix = day < lambda_new_samples[:,day]
    expected_texts_per_day[day] = (np.sum(lambda_1_samples[ix])+np.sum(lambda_2_samples[~ix]))/N

# Calculate confidence interals
upper_lim = expected_texts_per_day + 1.96*((3)**2/np.sqrt(N))
lower_lim = expected_texts_per_day - 1.96*((3)**2/np.sqrt(N))

# plot barplot of text data, expected texts per day with C.I., labels and colors
plt.figure(figsize = (15,10))
plt.plot(range(n_count_data), expected_texts_per_day, lw=5, color="#2D4571",label="$\lambda$(t)")
plt.plot(np.arange(0,len(expected_texts_per_day)),upper_lim,color = '#EF2976',lw = .3,label = "95% C.I. Upper Limit")
plt.plot(np.arange(0,len(expected_texts_per_day)),lower_lim,color = '#EF2976',lw = .3,label = "95% C.I. Lower Limit")
plt.xlim(0, n_count_data)
plt.xlabel("Day")
plt.ylabel("Number of Texts Sent")
plt.title("Expected Text Count")
plt.bar(np.arange(len(count_data)), count_data, color="#6F256F", alpha=0.65,label="observed text count")
plt.legend(loc="upper left");


# Plot histogram of final lamda samples
plt.hist(lambda_1_samples,alpha = .8,label = 'lambda_1')
plt.hist(lambda_2_samples,alpha = .8,label = 'lambda_2')
plt.legend(loc = "upper right")
plt.title("Lambda Distributions")
plt.show()