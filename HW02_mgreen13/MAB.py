import matplotlib.pyplot as plt
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



def simulate_function(armToPull,trueExpectedReward,metric,length,num_sims,ucb_bool):
    
    time = length
    num_runs = num_sims
    list_times = list(range(100,time,10))
    
    if ucb_bool == True:
        list_regrets_ucb = []
        list_regrets_e_first_greedy=[]
        
        for T in list_times:
            regret_e_first_g = np.mean([e_first_greedy(T,metric,armToPull,trueExpectedReward) for run in range(num_runs)])
            regret_ucb= np.mean([ucb(T,metric,armToPull,trueExpectedReward) for run in range(num_runs)])
            list_regrets_e_first_greedy.append(regret_e_first_g)
            list_regrets_ucb.append(regret_ucb)
        return(list_regrets_ucb,list_regrets_e_first_greedy)
        
    else:
        list_regrets_random = []
        list_regrets_greedy = []
        list_regrets_e_first_greedy = []
        list_regrets_e_greedy = []
        
        for T in list_times:
            regret_r = random(T,metric,armToPull,trueExpectedReward)
            regret_g = np.mean([greedy_update(T,metric,armToPull,trueExpectedReward) for run in range(num_runs)])
            regret_e_first_g = np.mean([e_first_greedy(T,metric,armToPull,trueExpectedReward) for run in range(num_runs)])
            regret_e_g = np.mean([e_greedy(T,.5,metric,armToPull,trueExpectedReward) for run in range(num_runs)])
            
            list_regrets_greedy.append(regret_g)
            list_regrets_random.append(regret_r)
            list_regrets_e_first_greedy.append(regret_e_first_g)
            list_regrets_e_greedy.append(regret_e_g)
        
        return(list_regrets_random,list_regrets_greedy,list_regrets_e_first_greedy,list_regrets_e_greedy)
    
def plotting_function(list_regrets_random,list_regrets_greedy,list_regrets_e_first_greedy,list_regrets_e_greedy,title,ylabel):
    X = list(range(0,1000,10))
    plt.figure(figsize = (10,7))
    plt.plot(X,list_regrets_e_first_greedy, color = 'green', label = 'e-first greedy')
    plt.plot(X,list_regrets_e_greedy, color = 'orange', label = 'e -reedy')
    plt.plot(X,list_regrets_greedy,color = "red",label='greedy')
    plt.plot(X,list_regrets_random,color = "black",label='random')
    plt.legend(loc = 'upper left')
    plt.title("{}".format(title))
    plt.xlabel("Gamble Duration")
    plt.ylabel("{}".format(ylabel))
    plt.show()


# --------------------------------------------- HARD BANDIT SIMULATIONS AND PLOTS ----------------------------------------
# DEFINE HARD BANDITE BETA DISTRIBUTIONS

centers = np.array([.2,.3,.4,.5,.6])
alphas,betas,hardTrueExpectedReward,hardArmToPull = dist_maker(700,centers,"Easy Bandit")

centers = np.array([.37,.4,.43,.46,.49])
alphas,betas,hardTrueExpectedReward,hardArmToPull = dist_maker(9,centers,"Hard Bandit")

##HARD BANDIT, 100 GAMBLES, REGRET,
list_regrets_random,list_regrets_greedy,list_regrets_e_first_greedy,list_regrets_e_greedy = simulate_function(hardArmToPull,hardTrueExpectedReward,True,1000,1,False)
plotting_function(list_regrets_random,list_regrets_greedy,list_regrets_e_first_greedy,list_regrets_e_greedy,"100 Gambles on 'Hard Bandit' Reward Distributions","Regret")
#
##HARD BANDIT, 100 GAMBLES, PERCENT
list_regrets_random,list_regrets_greedy,list_regrets_e_first_greedy,list_regrets_e_greedy, = simulate_function(hardArmToPull,hardTrueExpectedReward,False,1000,1,False)
plotting_function(list_regrets_random,list_regrets_greedy,list_regrets_e_first_greedy,list_regrets_e_greedy,"Percent of Optimal Pulls from Gambles on 'Hard Bandit'","% Pulls on Optimal Arm")

# EASY BANDIT, 100 GAMBLE, UCB and EPSILON FIRST GREEDY
list_regrets_ucb,list_regrets_e_first_greedy = simulate_function(hardArmToPull,hardTrueExpectedReward,True,1000,50,True)
np.save("ucb_easy",np.array(list_regrets_ucb))
# HARD BANDIT, 100 gambles, ucb and epsilon with %

list_regrets_ucb,list_regrets_e_first_greedy = simulate_function(hardArmToPull,hardTrueExpectedReward,True,1000,1,True)
list_regrets_ucbP,list_regrets_e_first_greedyP = simulate_function(hardArmToPull,hardTrueExpectedReward,False,1000,50,True)


# PLOT ucb vs ep on hard bandit
X = list(range(100,1000,10))
plt.figure(figsize = (10,7))
plt.plot(X,list_regrets_e_first_greedy, color = 'green', label = 'e-first greedy')
plt.plot(X,list_regrets_ucb,color = "red",label = "UCB")
plt.legend(loc = 'upper left')
plt.title("Average Regret of with Top Two Strategies Played on Hard Bandit")
plt.xlabel("Gamble Duration")
plt.ylabel("% Optimal Arm")
plt.show()


X = list(range(100,1000,10))
plt.figure(figsize = (10,7))
plt.plot(X,list_regrets_e_first_greedyP, color = 'green', label = 'e-first greedy')
plt.plot(X,list_regrets_ucbP,color = "red",label = "UCB")
plt.legend(loc = 'upper left')
plt.title("Percent of Optimal Pulls with Top Two Strategies Played on Hard Bandit")
plt.xlabel("Gamble Duration")
plt.ylabel("% Optimal Arm")
plt.show()

