import numpy as np
import matplotlib.pyplot as plt
from numba import jit,njit,prange
from pylab import rcParams


rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]=16
rcParams['axes.linewidth'] =1.5


## Prisoner's Dilemma=============================================================================================================
r=2.0
#===============================
R=r-1
S=-1
T=r
P=0.0

## Payoff function================================================================================================================

@njit
def payoff(action1,action2):
    if action1==1 and action2==1:
        z=R
    elif action1==1 and action2==0:
        z=S
    elif action1==0 and action2==1:
        z=T
    elif action1==0 and action2==0:
        z=P
    return z

## Likelihood====================================================================================================================

@njit
def likelihood(Evidence,sample_p,sample_q,coperation):
    if Evidence==1:
        z=sample_p*coperation + sample_q*(1-coperation)
    elif Evidence==0:
        z=(1-sample_p)*coperation + (1-sample_q)*(1-coperation)
    return z

## Marginal likelihood=============================================================================================================

@njit
def marginal_likelihood(Evidence,p,q,coperation,prior):
    r=0
    for i in range(len(p)):
          s=0
          for j in range(len(q)):
            sample_p=p[i]
            sample_q=q[j]
            z=prior[i][j]
            s+= likelihood(Evidence,sample_p,sample_q,coperation)*z
          r+=s
    return r

## Action for first iteration============================================================================================

@njit
def First_iteration_action(strategy):

    z=np.random.uniform(0,1)
    if z<strategy[2]:
        action=1
    else:
        action=0
    return action

## Module to define mechanism of the reactive strategy  =====================================================

@njit
def Action(strategy,last_action_opponent): 

    x=np.random.uniform(0,1)
    if last_action_opponent==1:
        if x<strategy[0]:
            action=1
        elif x>strategy[0]:
            action=0
    elif last_action_opponent==0: 
        if x<strategy[1]:
            action=1
        elif x>strategy[1]:
            action=0
    return action

## Find indexing of maximum element=============================================================================================

@njit
def max_post(arr2D):
    result = np.where(arr2D ==np.amax(arr2D))
    row=result[0]
    column=result[1]
    choice=np.random.choice(np.arange(0,len(row)))
    return row[choice], column[choice]

## Bayes update: Finding posterior and (p,q) where the posterior distribution becomes maximum=================================================

@njit
def Bayes_update(Evidence,p,q,coperation,prior):

    Posterior=np.zeros((len(p),len(q)))

    ## Calculationg posterior==========================

    for i in range(len(p)):
        for j in range(len(q)):
            sample_p=p[i]
            sample_q=q[j]
            z=prior[i][j]
            Posterior[i][j]=likelihood(Evidence,sample_p,sample_q,coperation)*z/marginal_likelihood(Evidence,p,q,coperation,prior)

    ## Find the maximum====================
    index_p,index_q=max_post(Posterior)
    pm,qm=p[index_p],q[index_q]
    return Posterior,pm,qm



#====================================================================================================================================

#One reactive player and one bayesian player ========================================================================================

#====================================================================================================================================


@njit      
def Reactive_Bayesian(Reactive_strategy,Bayesian_strategy,average,no_of_round):

    #Intialization of an array to append the action pairs=============
    action_pair=[]

    for e in range(average):
        ##Coping the strategy and use it for each realization from scratch============

        reactive_strategy=Reactive_strategy.copy()
        bayesian_strategy=Bayesian_strategy.copy()
        
        ##Initilazation of cooperation level ===================================

        cop_Bayesian=[]
        cop_Reactive=[]


        ## Initialization of prior distribution which is taken as uniform, (N,N) is the shape of distribution ===================
        
        prior_distribution=np.ones((N,N))
        
        #Starting the game =================================================

        for k in range(0,no_of_round):

            ## Store the cooperation level at each iteration=====================

            cop_Bayesian.append(bayesian_strategy[2])
            cop_Reactive.append(reactive_strategy[2])

            ##Choosing action==============================================

            if k==0:
                action_reactive=First_iteration_action(reactive_strategy)
                action_bayesian=First_iteration_action(bayesian_strategy)  

                ## Updating prior=====================================
                
                prior_distribution,bayesian_strategy[0],bayesian_strategy[1]=Bayes_update(action_reactive,p,q,bayesian_strategy[2],prior_distribution)
            
            else:
                prev_action_reactive=action_reactive
                prev_action_bayesian=action_bayesian
                action_reactive=Action(reactive_strategy,prev_action_bayesian)
                action_bayesian=Action(bayesian_strategy,prev_action_reactive)

                ## Updating prior=====================================

                prior_distribution,bayesian_strategy[0],bayesian_strategy[1]=Bayes_update(action_reactive,p,q,cop_Bayesian[k-1],prior_distribution)

            ## Append the action pairs into an array =================================
            if action_reactive==1 and action_bayesian==1:
                action_pair.append("CC")
            elif action_reactive==0 and action_bayesian==1:
                action_pair.append("DC")
            elif action_reactive==1 and action_bayesian==0:
                action_pair.append("CD")
            elif action_reactive==0 and action_bayesian==0:
                action_pair.append("DD")
            
            ##Calculating cooperation level based on true strategy of player==============

            reactive_strategy[2]=Reactive_strategy[0]*cop_Bayesian[k]+ Reactive_strategy[1]*(1-cop_Bayesian[k])
            bayesian_strategy[2]=bayesian_strategy[0]*cop_Reactive[k]+bayesian_strategy[1]*(1-cop_Reactive[k])

    number_of_CC=action_pair.count("CC")/average
    number_of_CD=action_pair.count("CD")/average
    number_of_DC=action_pair.count("DC")/average
    number_of_DD=action_pair.count("DD")/average
    actions_count={"CC": number_of_CC, "CD": number_of_CD, "DC": number_of_DC, "DD": number_of_DD}

    return actions_count


##Parameters=======================================================================================================================

no_of_round=700
average=10000
discount_factor=0.75
N=11
p=q=np.linspace(0,1,N)
Reactive_strategy=np.array([0.5, 0.6, 0.5])
Bayesian_strategy=np.array([0.5, 0.6, 0.5])

## ================================================================================================================================

actions_count=Reactive_Bayesian(Reactive_strategy,Bayesian_strategy,average,no_of_round)
actions=np.array(list(actions_count.keys()))
values=np.array(list(actions_count.values()))
np.savetxt('Fig8(a)_data.txt', values)


## Plot===========================================================================================================================
yticks=[0.0,0.25,0.5]
bars=plt.bar(actions, values/no_of_round, width = 0.3)
bars[0].set_color('green')## CC
bars[1].set_color('orange') ## CD
bars[2].set_color('#03a1fc') ## DC
bars[3].set_color('red') ## DD
plt.yticks(yticks, fontsize=28)
plt.xticks(actions, fontsize=28)
plt.savefig("Fig8(a).png",dpi=600)
plt.show()
