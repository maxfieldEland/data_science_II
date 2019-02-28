# -*- coding: utf-8 -*-
"""
Created on Sun Feb  3 13:57:55 2019

@author: mgreen13
"""

import matplotlib.pyplot as plt
import random
import operator
import scipy.stats as stats
import numpy as np

 =============================================================================
 armToPull = {}
 armTrueExpectedReward = {}
 arm = 0
 b = 9
 for a in [2,3,4,5,6]:
     b = 8 -a
     p = stats.beta(a,b)
     armToPull[arm] = p
     armTrueExpectedReward[arm] = a/(a+b)
     arm += 1
     
 arms = list(armToPull.keys())
 #plot distributions for each of the arms
 x = np.linspace(0,1,100)
 for arm in arms:
     p = armToPull[arm]
     a,b = p.args
     plt.plot(x,p.pdf(x), label="arm {}: $B(a = {}, b = {})$".format(arm,a,b))
 
 =============================================================================

 =============================================================================
 # Shrink current axis by 20% to make room for legend:
 ax = plt.gca()
 box = ax.get_position()
 ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
 
 # Put a legend to the right of the current axis
 ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
 
 plt.xlabel("reward")
 plt.ylabel("Prob. density")
 #
 =============================================================================

# ----------------------------------------- Homebrewed functions used throughout the notebook --------------------------------
def dist_maker(a,centers,title):
    """Helper function to quickly visualize beta function pdfs given a constant alpha and a list of desired centers """
    betas = []
    for i in centers:
        betas.append((a - i*a)/i)

    
    armToPull = {}
    arm = 0
    for b in betas:
        p = stats.beta(a,b)
        armToPull[arm] = p
        arm += 1
    arms = list(armToPull.keys())
    #plot distributions for each of the arms
    x = np.linspace(0,1,100)
    for arm in arms:
        p = armToPull[arm]
        a,b = p.args
        plt.plot(x,p.pdf(x), label="$B(a = {}, b = {})$".format(a,np.round(b,2)))
    # Shrink current axis by 20% to make room for legend:
    ax = plt.gca()
    box = ax.get_position()
    ax.set_position([box.x0,box.y0,box.width*0.8,box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
    plt.xlabel("Reward")
    plt.ylabel("Prob. Density")
    plt.title("{}".format(title))
    
    return(np.array([a]*5),betas)
    
    


    

def play_arm(i):
    return armToPull[i].rvs()



def expected_regret_new(time,rewards):
    regret = (time)*max(armTrueExpectedReward.values()) - rewards
    return(regret)
     


# part two 
# implementing random algorithm

def random_pulls(time,reg_boolean):
    regret = np.zeros([time])
    reward = np.zeros([time])
    pull_hist = np.zeros([5])
    for idx,t in enumerate(list(range(time))):
        i = np.random.randint(0,5)
        pull_hist[i] += 1
        reward[idx] = play_arm(i) 
        # are we using regret as a metric or not?
        if reg_boolean == True:
            regret[idx] = expected_regret_new(t,sum(reward))
    # if not, than return the fraction of pulls on best arm 
    if reg_boolean == False:
        return(pull_hist[4]/t)
    else:
        return(regret[-1])
    
    


time = 1000
random_reg = np.zeros([int(time/10)-10])
greedy_reg = np.zeros([int(time/10)-10])
ep_first_greedy_reg = np.zeros([int(time/10)-10])
ep_greedy_reg = np.zeros([int(time/10)-10])
#ucb_reg = np.zeros([int(time/10)-10])
rewards = []
for idx,gamble in enumerate(list(range(100,time,10))):
   random_reg[idx] = random_pulls(gamble, True)
   #ucb_reg[idx] = ucb(gamble)
   #greedy_reg[idx] = greedy(gamble,False)
   ep_first_greedy_reg[idx] = ep_first_greedy(gamble,10,True)
   ep_greedy_reg[idx] = ep_greedy(gamble,.2,False)



plt.figure(figsize = (10,8))
plt.scatter(list(range(len(ep_first_greedy_reg))),ep_first_greedy_reg, color = 'green', label = 'e-first greedy')
plt.scatter(list(range(len(ep_greedy_reg))),ep_greedy_reg, color = 'orange', label = 'e greedy')

#plt.plot(list(range(len(ucb_reg))),ucb_reg,color = "blue",label='ucb')
plt.scatter(list(range(len(random_reg))),random_reg,color = "black",label='random')
#plt.scatter(list(range(len(greedy_reg))),greedy_reg, color = "red",label = "greedy")
plt.legend(loc = 'upper left')



def greedy(time,reg_boolean):
    # play each arm once to find initial estimate r^hat_i
    rHat_i = {0:1,1:1,2:1,3:1,4:1}
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    regrets = []
    
    for i in range(5):
        r = play_arm(i)
        reward_hist[i].append(r)
        rHat_i[i] = r
        regrets.append(expected_regret_new(i+1,sum(reward_hist[i])))  
    i_plus = max(rHat_i,key = rHat_i.get)
    
    for t in range(5,time):
        r = play_arm(i_plus)
        reward_hist[i_plus].append(r)
        regrets.append(expected_regret_new(t,sum(reward_hist[i_plus])))
        
    if reg_boolean == False:
        return(len(reward_hist[4])/t)
    else:
        return(regrets[-1])
        
    
    
def ep_first_greedy(time,m,reg_boolean):
    # play each arm once to find initial estimate r^hat_i
    # Does e first greedy exploration count towards total reward hist
    rHat_i = {0:1,1:1,2:2,3:3,4:4}
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    regrets = []
    # Exploration Phase
    for j in range(m):
        for i in range(5):
            r = play_arm(i)
            reward_hist[i].append(r)
            n = len(reward_hist[i])
            rHat_i[i] = (1/n)*sum(reward_hist[i])
            summed_rewards = []
            # unravel 
            for r in reward_hist.values():
                summed_rewards.extend(r)
            regrets.append(expected_regret_new(j,sum(summed_rewards)))
    # Exploitation Phase
    exploit_start_time = 5*m
    for t in range(exploit_start_time,time):
        i_plus = max(rHat_i,key = rHat_i.get)
        r = play_arm(i_plus)
        # add reward generated to reward_hist
        reward_hist[i_plus].append(r)
        # Update regret history
        regrets.append(expected_regret_new(t,sum(reward_hist[i_plus])))
    if reg_boolean == False:
        return(len(reward_hist[4])/t)
    else:
        return(regrets[-1])
    
        
def ep_greedy(time,e,reg_boolean):
    
    # play each arm once to find initial estimate r^hat_i
    rHat_i = {0:1,1:1,2:2,3:3,4:4}
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    regrets = []
    # Exploration Phase
    for t in range(time):
        rand  = np.random.rand()
        # pick random with chance e, else pick the arm with the best average rewards 
        if e > rand:
            i_plus = np.random.randint(0,5)
        else:
            i_plus = max(rHat_i,key = rHat_i.get)
        r = play_arm(i_plus)
        # add reward generated to reward_hist
        reward_hist[i_plus].append(r)
        # Update regret history
        if reg_boolean == True:
            regrets.append(expected_regret_new(t,sum(reward_hist[i_plus])))
        
    if reg_boolean == False:
        return(len(reward_hist[4])/t)
    else:
        return(regrets[-1])

def ucb(time,reg_boolean):
    # play each arm once to find initial estimate r^hat_i
    # Does e first greedy exploration count towards total reward hist
    q = {0:1,1:1,2:2,3:3,4:4}
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    regrets = []
    # PLay each arm once
    for i in range(5):
        r = play_arm(i)
        reward_hist[i].append(r)
        regrets.append(expected_regret_new(i,sum(reward_hist[i])))
        q[i] = np.mean(reward_hist[i])+np.sqrt(2*np.log(i+1)/len(reward_hist[i]))
    # begin greedy section
    for t in range(i,time):
        i_plus = max(q,key = q.get)
        r = play_arm(i_plus)
        # add reward generated to reward_hist
        reward_hist[i].append(r)
        # Update regret history
        regrets.append(expected_regret_new(i_plus,sum(reward_hist[i_plus])))
        # update q
        q[i_plus] = np.mean(reward_hist[i_plus])+np.sqrt(2*np.log(t)/len(reward_hist[i_plus]))
    if reg_boolean == False:
        return(len(reward_hist[4])/t)
    else:
        return(regrets[-1])
    
    
time = 1000
random_reg = np.zeros([int(time/10)-10])
#greedy_reg = np.zeros([int(time/10)-10])
#ep_first_greedy_reg = np.zeros([int(time/10)-10])
#ep_greedy_reg = np.zeros([int(time/10)-10])
#ucb_reg = np.zeros([int(time/10)-10])
rewards = []
for idx,gamble in enumerate(list(range(100,time,10))):
   random_reg[idx] = random_pulls(gamble, False)
   #ucb_reg[idx] = ucb(gamble)
 #  greedy_reg[idx] = greedy(gamble)
  # ep_first_greedy_reg[idx] = ep_first_greedy(gamble,10)
   #ep_greedy_reg[idx] = ep_greedy(gamble,.2)



plt.figure(figsize = (10,8))
#plt.plot(list(range(len(ep_first_greedy_reg))),ep_first_greedy_reg, color = 'green', label = 'e-first greedy')
#plt.plot(list(range(len(ucb_reg))),ucb_reg,color = "blue",label='ucb')
plt.plot(list(range(len(random_reg))),random_reg,color = "black",label='random')
plt.plot(list(range(len(greedy_reg))),greedy_reg, color = "red",label = "greedy")
plt.legend(loc = 'upper left')


# problem 3
num_runs = 10
list_times = range(100,1000,10)
list_regrets_random = []
list_regrets_greedy = []
list_regrets_e_first_greedy = []
list_regrets_e_greedy = []
list_regrets_ucb = []

for T in list_times:
    regret_r = random_pulls(T,False)
    regret_g = np.mean([greedy(T,False) for run in range(num_runs)])
    regret_e_first_g = np.mean([ep_first_greedy(T,10,False) for run in range(num_runs)])
    regret_e_g = np.mean([ep_greedy(T,.5,False) for run in range(num_runs)])
    #regret_ucb = np.mean([ucb(gamble) for run in range(num_runs)])
    
    list_regrets_greedy.append(regret_g)
    list_regrets_random.append(regret_r)
    list_regrets_e_first_greedy.append(regret_e_first_g)
    list_regrets_e_greedy.append(regret_e_g)
    #list_regrets_ucb.append(regret_ucb)
    
plt.figure(figsize = (10,7))
plt.plot(list(range(len(ep_first_greedy_reg))),ep_first_greedy_reg, color = 'green', label = 'e-first greedy')
plt.plot(list(range(len(ep_greedy_reg))),ep_greedy_reg, color = 'orange', label = 'e greedy')

plt.plot(list(range(len(list_regrets_greedy))),list_regrets_greedy,color = "blue",label='greedy')
plt.plot(list(range(len(list_regrets_random))),list_regrets_random,color = "black",label='random')
plt.plot(list(range(len(list_regrets_ucb))),list_regrets_ucb,color = "red",label='ucb')

#plt.plot(list(range(len(greedy_reg))),greedy_reg, color = "red",label = "greedy")
plt.legend(loc = 'upper left')



     
#---------------------------------------- Problem 3 Easy and Hard Bandits ------------------

# ---------------------------------------------- hard bandit ------------------------------------
centers = np.array([.3,.4,.5,.6,.7])
alphas,betas = dist_maker(3,centers,"Hard Bandit")

armToPull = {}
armTrueExpectedReward = {}
arm = 0

for idx,a in enumerate(alphas):
    
    p = stats.beta(a,betas[idx])
    armToPull[arm] = p
    armTrueExpectedReward[arm] = a/(a+betas[idx])
    arm += 1
    
arms = list(armToPull.keys())
#plot distributions for each of the arms
x = np.linspace(0,1,100)
for arm in arms:
    p = armToPull[arm]
    a,b = p.args
    plt.plot(x,p.pdf(x), label="arm {}: $B(a = {}, b = {})$".format(arm,a,np.round(b,2)))


# Shrink current axis by 20% to make room for legend:

# Put a legend to the right of the current axis


time = 1000
random_reg = np.zeros([int(time/10)-10])
greedy_reg = np.zeros([int(time/10)-10])
ep_first_greedy_reg = np.zeros([int(time/10)-10])
ep_greedy_reg = np.zeros([int(time/10)-10])
rewards = []

for idx,gamble in enumerate(list(range(100,time,10))):
   random_reg[idx] = random_pulls(gamble)
   greedy_reg[idx] = greedy(gamble)
   ep_first_greedy_reg[idx] = ep_first_greedy(gamble,10)
   ep_greedy_reg[idx] = ep_greedy(gamble,.2)

plt.figure(figsize = (10,8))
plt.plot(list(range(len(ep_first_greedy_reg))),ep_first_greedy_reg, color = 'green', label = 'e-first greedy')
plt.plot(list(range(len(random_reg))),random_reg,color = "black",label='random')
plt.plot(list(range(len(ep_greedy_reg))),ep_greedy_reg,color = "blue",label='ep-greedy')

plt.plot(list(range(len(greedy_reg))),greedy_reg, color = "red",label = "greedy")
plt.legend(loc = 'upper left')


# -------------------------------------- EASY BANDIT --------------------------
centers = np.array([.3,.4,.5,.6,.7])
alphas,betas = dist_maker(140,centers,"Easy Bandit")


# -------------------------------------------- trials of one gamble ----------------------------

time = 1000
random_reg = np.zeros([int(time/10)-10])
greedy_reg = np.zeros([int(time/10)-10])
ep_first_greedy_reg = np.zeros([int(time/10)-10])
ep_greedy_reg = np.zeros([int(time/10)-10])
rewards = []

for idx,gamble in enumerate(list(range(100,time,10))):
   random_reg[idx] = random_pulls(gamble)
   greedy_reg[idx] = greedy(gamble)
   #ep_first_greedy_reg[idx] = ep_first_greedy(gamble,10)
   #ep_greedy_reg[idx] = ep_greedy(gamble,.2)

plt.figure(figsize = (10,8))
plt.plot(list(range(len(ep_first_greedy_reg))),ep_first_greedy_reg, color = 'green', label = 'e-first greedy')
plt.plot(list(range(len(random_reg))),random_reg,color = "black",label='random')
#plt.plot(list(range(len(ep_greedy_reg))),ep_greedy_reg,color = "blue",label='ep-greedy')

plt.plot(list(range(len(greedy_reg))),greedy_reg, color = "red",label = "greedy")
plt.legend(loc = 'upper left')

# -------------------------------------------- trials averaged over several gambles ----------------------------

num_runs = 10
list_times = range(100,1000,10)
list_regrets_random = []
list_regrets_greedy = []
list_regrets_e_first_greedy = []
list_regrets_e_greedy = []

for T in list_times:
    regret_r = random_pulls(T)
    regret_g = np.mean([greedy(T) for run in range(num_runs)])
    regret_e_first_g = np.mean([ep_first_greedy(T,10) for run in range(num_runs)])
    regret_e_g = np.mean([ep_greedy(T,.5) for run in range(num_runs)])
    
    list_regrets_greedy.append(regret_g)
    list_regrets_random.append(regret_r)
    list_regrets_e_first_greedy.append(regret_e_first_g)
    list_regrets_e_greedy.append(regret_e_g)
    
plt.figure(figsize = (10,7))
plt.plot(list(range(len(ep_first_greedy_reg))),ep_first_greedy_reg, color = 'green', label = 'e-first greedy')
plt.plot(list(range(len(ep_greedy_reg))),ep_greedy_reg, color = 'orange', label = 'e greedy')
plt.plot(list(range(len(list_regrets_greedy))),list_regrets_greedy,color = "blue",label='greedy')
plt.plot(list(range(len(list_regrets_random))),list_regrets_random,color = "black",label='random')

plt.legend(loc = 'upper left')

# ----------------------------------------- use of new metric------------------------------------------
num_runs = 10
list_times = range(100,1000,10)
list_regrets_random = []
list_regrets_greedy = []
list_regrets_e_first_greedy = []
list_regrets_e_greedy = []

for T in list_times:
    regret_r = random_pulls(T)
    regret_g = np.mean([greedy(T) for run in range(num_runs)])
    regret_e_first_g = np.mean([ep_first_greedy(T,10) for run in range(num_runs)])
    regret_e_g = np.mean([ep_greedy(T,.5) for run in range(num_runs)])
    
    list_regrets_greedy.append(regret_g)
    list_regrets_random.append(regret_r)
    list_regrets_e_first_greedy.append(regret_e_first_g)
    list_regrets_e_greedy.append(regret_e_g)
    
plt.figure(figsize = (10,7))
plt.plot(list(range(len(ep_first_greedy_reg))),ep_first_greedy_reg, color = 'green', label = 'e-first greedy')
plt.plot(list(range(len(ep_greedy_reg))),ep_greedy_reg, color = 'orange', label = 'e greedy')
plt.plot(list(range(len(list_regrets_greedy))),list_regrets_greedy,color = "blue",label='greedy')
plt.plot(list(range(len(list_regrets_random))),list_regrets_random,color = "black",label='random')

plt.legend(loc = 'upper left')
