# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:17:04 2019

@author: mgreen13
"""
import matplotlib.pyplot as plt
import operator
import scipy.stats as stats
import numpy as np          


# --------------------- FUNCTIONS USED -----------------
def dist_maker(a,centers,title):
    """Helper function to quickly visualize beta function pdfs given a constant alpha and a list of desired centers """
    betas = []
    for i in centers:
        betas.append((a - i*a)/i)
    
    armTrueExpectedReward = {}
    armToPull = {}
    arm = 0
    for b in betas:
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
    
    return(np.array([a]*5),betas,armTrueExpectedReward,armToPull)


# ---------------------------- Problem One -------------------------

def expected_regret_new(t,rewards,armTrueExpectedReward):
    regret = (t)*max(armTrueExpectedReward.values()) - rewards
    return(regret)
     
# --------------------------- Problem Two ----------------------
# Below are algorithms that work to maximize return on gambles
# Algorithms are designed to keep track of either regrets or the percent of best arms being pulled.
    
def random(time,reg_boolean,armToPull,armTrueExpectedReward):
    regret = np.zeros([time])
    reward = np.zeros([time])
    pull_hist = np.zeros([5])
    for idx,t in enumerate(list(range(time))):
        i = np.random.randint(0,5)
        pull_hist[i] += 1
        # pull arm
        reward[idx] = armToPull[i].rvs()

    # if not, than return the fraction of pulls on best arm 
    if reg_boolean == False:
        return(pull_hist[4]/time)
    else:
        regret = expected_regret_new(time,np.sum(np.sum(reward)),armTrueExpectedReward)
        return(regret)

        
        
def greedy_update(time,reg_boolean,armToPull,armTrueExpectedReward):
    # play each arm once to find initial estimate r^hat_i
    rHat_i = [1/2,1/2,1/2,1/2,1/2]
    rewards = np.zeros([time])
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    #initially pull random arm
    i= np.random.randint(5)
    # pull that arm for the rest of time
    for t in range(time):
        r = armToPull[i].rvs()
        reward_hist[i].append(r)
        i = np.argmax(rHat_i)
        rHat_i[i] = np.mean(reward_hist[i])
        rewards[t] = r
    if reg_boolean == False:
        return(len(reward_hist[4])/time)
    else:
        regret = expected_regret_new(time,np.sum(rewards),armTrueExpectedReward)
        return(regret)

    

def e_first_greedy(time, reg_boolean,armToPull,armTrueExpectedReward):
    rHat_i = [1,1,1,1,1]
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    t = 0
    rewards = np.zeros([time])
    m = int(2*np.log(time*5))
    # EXPLORATION PHASE
    for j in range(m):
        # pull every arm
        for i in range(5):
            r = armToPull[i].rvs()
            reward_hist[i].append(r)
            rHat_i[i] = np.mean(reward_hist[i])
            rewards[t] = r
            t = t+1
        
    i = np.argmax(rHat_i)

    # EXPLOITATION PHASE
    for t in range(5*m, time):
        r = armToPull[i].rvs()
        reward_hist[i].append(r)
        rewards[t] = r
        
    
    if reg_boolean == False:
        return(len(reward_hist[4])/time)
    else:
        regret = expected_regret_new(time,np.sum(rewards),armTrueExpectedReward)
        return(regret)


def e_greedy(time, e, reg_boolean,armToPull,armTrueExpectedReward):
    rHat_i = [1/2,1/2,1/2,1/2,1/2]
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    rewards = np.zeros([time])
    for t in range(time):
        rand = np.random.rand()
        # pick random arm with chance e
        if rand > e:
            i = np.argmax(rHat_i)
        else:
            i = np.random.randint(0,5)
        
        r = armToPull[i].rvs()
        rHat_i[i] = np.mean(reward_hist[i])
        reward_hist[i].append(r)
        rewards[t] = r
        
        
    if reg_boolean == False:
        return(len(reward_hist[4])/time)
    else:
        regret = expected_regret_new(time,np.sum(rewards),armTrueExpectedReward)
        return(regret)


# create ucb1 algorithm
def ucb(time, reg_boolean,armToPull,armTrueExpectedReward):
    q = [1,1,1,1,1]
    reward_hist = {0:[],1:[],2:[],3:[],4:[]}
    regrets = []
    i = np.random.randint(5)
    for t in range(time):
        r = armToPull[i].rvs()
        reward_hist[i].append(r)
        regrets.append(expected_regret_new(t,np.sum(reward_hist[i]),armTrueExpectedReward))
        q[i]=np.mean(reward_hist[i])+np.sqrt((2*np.log(t+1))/len(reward_hist[i]))
        i = np.argmax(q)

    if reg_boolean == False:
        return(len(reward_hist[4])/time)
    else:
        return(regrets[-1])

# ---------------------------- TESTING PHASE ---------------------

# gamble with easy bandit

centers = np.array([.1,.3,.5,.7,.9])
alphas,betas,hardTrueExpectedReward,hardArmToPull = dist_maker(900,centers,"Easy Bandit")

time = 1000
random_reg = np.zeros([int(time/10)-10])
greedy_reg = np.zeros([int(time/10)-10])
e_first_greedy_reg = np.zeros([int(time/10)-10])
e_greedy_reg = np.zeros([int(time/10)-10])
ucb_reg = np.zeros([int(time/10)-10])

rewards = []

for idx,gamble_duration in enumerate(list(range(100,time,10))):
   random_reg[idx] = random(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   #ucb_reg[idx] = ucb(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   greedy_reg[idx] = greedy_update(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   e_first_greedy_reg[idx] = e_first_greedy(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   e_greedy_reg[idx] = e_greedy(gamble_duration,.1,True,hardArmToPull,hardTrueExpectedReward)


plt.figure(figsize = (15,10))
plt.plot(list(range(100,1000,10)),e_first_greedy_reg, color = 'green', label = 'e-first greedy')
plt.plot(list(range(100,1000,10)),e_greedy_reg,color = 'orange', label = 'e greedy')
#plt.plot(list(range(100,time,10)),ucb_reg,color = "blue",label='ucb')
plt.plot(list(range(100,time,10)),random_reg,color = "black",label='random')
plt.plot(list(range(100,1000,10)),greedy_reg, color = "red",label = "greedy")
plt.legend(loc = 'upper left')
plt.title("Regret over 100 Gambles up to T = 1000 with Easy ard Bandit")
plt.xlabel("Gamble Duration")
plt.ylabel("Regret")



# Simulate over 100 trials with easy bandit
num_runs = 100
list_times = range(100,1000,10)
list_regrets_random = []
list_regrets_greedy = []
list_regrets_e_first_greedy = []
list_regrets_e_greedy = []
list_regrets_ucb = []

for T in list_times:
    regret_r = random(T,True,hardArmToPull,hardTrueExpectedReward)
    regret_g = np.mean([greedy_update(T,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
    regret_e_first_g = np.mean([e_first_greedy(T,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
    regret_e_g = np.mean([e_greedy(T,.1,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
    #regret_ucb = np.mean([ucb(gamble,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
    
    list_regrets_greedy.append(regret_g)
    list_regrets_random.append(regret_r)
    list_regrets_e_first_greedy.append(regret_e_first_g)
    list_regrets_e_greedy.append(regret_e_g)
    #list_regrets_ucb.append(regret_ucb)
    
    
plt.figure(figsize = (15,10))
plt.plot(list(range(100,1000,10)),list_regrets_e_first_greedy, color = 'green', label = 'e-first greedy ({} gambles)'.format(num_runs))
plt.plot(list(range(100,1000,10)),list_regrets_e_greedy, color = 'orange', label = 'e greedy ({} gambles)'.format(num_runs))
plt.plot(list(range(100,1000,10)),list_regrets_greedy,color = "red",label='greedy ({} gambles)'.format(num_runs))
plt.plot(list(range(100,1000,10)),list_regrets_random,color = "black",label='random')
#plt.plot(list(range(100,1000,10)),list_regrets_ucb,color = "purple",label='ucb')

plt.legend(loc = 'upper left')
plt.xlabel("Gamble Duration(T)")
plt.ylabel("Regret")

# ====================================================== |||||||||||| ==================================
# ---------------------------------------------------HARD bandit-----------------------------
# ====================================================== |||||||||||| ==================================



X = list(range(100,1000,10))

centers = np.array([.37,.4,.43,.46,.49])
alphas,betas,hardTrueExpectedReward,hardArmToPull = dist_maker(9,centers,"Hard Bandit")


time = 1000
random_reg = np.zeros([int(time/10)-10])
greedy_reg = np.zeros([int(time/10)-10])
e_first_greedy_reg = np.zeros([int(time/10)-10])
e_greedy_reg = np.zeros([int(time/10)-10])
ucb_reg = np.zeros([int(time/10)-10])


for idx,gamble_duration in enumerate(list(range(100,time,10))):
   m = int(2*np.log(5*gamble_duration))
   random_reg[idx] = random(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   #ucb_reg[idx] = ucb(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   greedy_reg[idx] = greedy_update(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   e_first_greedy_reg[idx] = e_first_greedy(gamble_duration,True,hardArmToPull,hardTrueExpectedReward)
   e_greedy_reg[idx] = e_greedy(gamble_duration,.1,True,hardArmToPull,hardTrueExpectedReward)



plt.figure(figsize = (10,8))

plt.plot(list(range(100,1000,10)),list_regrets_e_first_greedy, color = 'green', label = 'e-first greedy ({} gambles)'.format(num_runs))
plt.plot(list(range(100,1000,10)),list_regrets_e_greedy, color = 'orange', label = 'e greedy ({} gambles)'.format(num_runs))
plt.plot(list(range(100,1000,10)),list_regrets_greedy,color = "red",label='greedy ({} gambles)'.format(num_runs))
plt.plot(list(range(100,1000,10)),list_regrets_random,color = "black",label='random')
#plt.plot(list(range(100,1000,10)),list_regrets_ucb,color = "purple",label='ucb')

# ----------------------------------------------------- CALCULATE AVERAGE PERCENT METRIC WITH HARD ---------------------------------------------

centers = np.array([.37,.4,.43,.46,.49])
alphas,betas,hardTrueExpectedReward,hardArmToPull = dist_maker(9,centers,"Hard Bandit")

time = 1000
num_runs = 100
list_times = range(100,time,10)
list_regrets_randomPH = []
list_regrets_greedyPH = []
list_regrets_e_first_greedyPH = []
list_regrets_e_greedypH = []
list_regrets_ucbPH= []

for T in list_times:
    regret_r = random(T,False,hardArmToPull,hardTrueExpectedReward)
    regret_g = np.mean([greedy_update(T,False,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])\
    
    regret_e_first_g = np.mean([e_first_greedy(T,False,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
    
    regret_e_g = np.mean([e_greedy(T,.1,False,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
    regret_ucb = np.mean([ucb(gamble,False,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
    
    list_regrets_greedyPH.append(regret_g)
    list_regrets_randomPH.append(regret_r)
    list_regrets_e_first_greedyPH.append(regret_e_first_g)
    list_regrets_e_greedyPH.append(regret_e_g)
    list_regrets_ucbPH.append(regret_ucb)
    
plt.figure(figsize = (10,7))
plt.plot(X,list_regrets_e_first_greedyPH, color = 'green', label = 'e-first greedy')
plt.plot(X,list_regrets_e_greedyPH, color = 'orange', label = 'e -reedy')
plt.plot(X,list_regrets_greedyPH,color = "red",label='greedy')
plt.plot(X,list_regrets_randomPH,color = "black",label='random')
plt.title("Percent of Gamble on Optimal Arm with Hard Bandit")
plt.xlabel("Duration of Gamble")
plt.ylabel("Percent of Gamble on Optimal Arm")
plt.legend(loc = "upper left")

# -------------------------------------------------- Average over 100 runs on hard bandit ----------------------------------------------
# Simulate over 100 trials with hard bandit
def plot_hard_norm_avg():
    time = 1000
    num_runs = 2
    list_times = range(100,time,10)
    list_regrets_randomH = []
    list_regrets_greedyH = []
    list_regrets_e_first_greedyH = []
    list_regrets_e_greedyH = []
    list_regrets_ucbH = []
    
    for T in list_times:
        regret_r = random(T,True,hardArmToPull,hardTrueExpectedReward)
        regret_g = np.mean([greedy_update(T,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
        regret_e_first_g = np.mean([e_first_greedy(T,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
        regret_e_g = np.mean([e_greedy(T,.5,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
        regret_ucb = np.mean([ucb(T,True,hardArmToPull,hardTrueExpectedReward) for run in range(num_runs)])
        
        list_regrets_greedyH.append(regret_g)
        list_regrets_randomH.append(regret_r)
        list_regrets_e_first_greedyH.append(regret_e_first_g)
        list_regrets_e_greedyH.append(regret_e_g)
        list_regrets_ucbH.append(regret_ucb)
        
    plt.figure(figsize = (10,7))
    plt.plot(X,list_regrets_e_first_greedyH, color = 'green', label = 'e-first greedy')
    plt.plot(X,list_regrets_e_greedyH, color = 'orange', label = 'e -reedy')
    plt.plot(X,list_regrets_greedyH,color = "red",label='greedy')
    plt.plot(X,list_regrets_randomH,color = "black",label='random')
    plt.legend(loc = 'upper left')
    plt.title("Regret over 100 Gambles up to T = 1000 with Hard Bandit")
    plt.xlabel("Gamble Duration")
    plt.ylabel("Regret")
    plt.show()

plot_hard_norm_avg()

    
    
