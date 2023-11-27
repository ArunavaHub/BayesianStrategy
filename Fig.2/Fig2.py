
import numpy as np
from numba import jit,njit,prange
import matplotlib.pyplot as plt
from pylab import rcParams

## Setting the style of the font==================================================================================
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]=16
rcParams['axes.linewidth'] =1.5



## Defining likelihood ============================================================================================
@njit
def likelihood(Evidence,sample_p,sample_q,coperation):
    if Evidence==1:
        z=sample_p*coperation + sample_q*(1-coperation)
    elif Evidence==0:
        z=(1-sample_p)*coperation + (1-sample_q)*(1-coperation)
    return z

## Marginal likelihood=============================================================================================

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

## Action for first iteration=====================================================================================

@njit
def First_iteration_action(strategy):

    z=np.random.uniform(0,1)
    if z<strategy[2]:
        action=1
    else:
        action=0
    return action

## Module to define the mechanism of reactive strategy  ===========================================================

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

## Find indexing of the maximum element of a matrix====================================================================

@njit
def max_post(arr2D):
    result = np.where(arr2D ==np.amax(arr2D))
    row=result[0]
    column=result[1]
    choice=np.random.choice(np.arange(0,len(row)))
    return row[choice], column[choice]

## Bayes Update: Finding Posterior and (p,q) where the posterior distribution becomes maximum=================================================

@njit
def Bayes_update(Evidence,p,q,coperation,prior):

    Posterior=np.zeros((len(p),len(q)))

    ## Calculationg Posterior==========================

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

## Parameters=======================================================================================================================

no_of_round=1001
N=51
average=1
p=q=np.linspace(0,1,N)

## Initialization of true strategies====================================================================================
Reactive_strategy=np.array([0.8,0.3,0.5])
Bayesian_strategy=np.array([0.8,0.3,0.5])

for e in range(average):

    ##Coping the strategy and use it for each realization from scratch============================================

    reactive_strategy=Reactive_strategy.copy()
    bayesian_strategy=Bayesian_strategy.copy()

    ##Initialzation of two arrays to store the updated (p,q) of Bayesian player==================================

    Bayesian_p=[]
    Bayesian_q=[]

    ##Initilazation of cooperation level ===================================

    cop_Bayesian=[]
    cop_Reactive=[]

    ## Initialization of prior distribution which is taken as a (N,N) shape uniform distribution ========
    
    prior_distribution=np.ones((N,N))

    #Starting the game ===========================================================================================

    for k in range(0,no_of_round):

        ## Store the cooperation level at each iteration==========================================================

        cop_Bayesian.append(bayesian_strategy[2])
        cop_Reactive.append(reactive_strategy[2])

        ##Choosing action=========================================================================================

        if k==0:
            ## First round========================================================================================
            action_reactive=First_iteration_action(reactive_strategy)
            action_bayesian=First_iteration_action(bayesian_strategy)  

            ## Updating prior=====================================
            
            prior_distribution,bayesian_strategy[0],bayesian_strategy[1]=Bayes_update(action_reactive,p,q,bayesian_strategy[2],prior_distribution)
        
        else:
            ## Subsequent rounds==================================================================================
            prev_action_reactive=action_reactive
            prev_action_bayesian=action_bayesian

            action_reactive=Action(reactive_strategy,prev_action_bayesian)
            action_bayesian=Action(bayesian_strategy,prev_action_reactive)

            ## Updating prior=====================================
            prior_distribution,bayesian_strategy[0],bayesian_strategy[1]=Bayes_update(action_reactive,p,q,cop_Bayesian[k-1],prior_distribution)
            
        ##Calculating cooperation level based on true strategy of player==============

        reactive_strategy[2]=Reactive_strategy[0]*cop_Bayesian[k]+ Reactive_strategy[1]*(1-cop_Bayesian[k])
        bayesian_strategy[2]=bayesian_strategy[0]*cop_Reactive[k]+bayesian_strategy[1]*(1-cop_Reactive[k])

        if k==50:
            np.savetxt('Fig2_50th_round_(p, q)=(0.8, 0.3).txt',prior_distribution)
        if k==500:
            np.savetxt('Fig2_500th_round_(p, q)=(0.8, 0.3).txt',prior_distribution)



## Import the data from the file==================================================================================

Posterior=np.loadtxt("Fig2_50th_round_(p, q)=(0.8, 0.3).txt")
Posterior=np.transpose(Posterior) ## Make the q along y axis and p along x axis===================================

#Draw the figure===============================================================================================

xticks=[0.0,0.25,0.5,0.75,1.0] 
plt.contourf(q,p,Posterior, cmap='plasma')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=22) 
plt.xticks(xticks, fontsize=24)
plt.yticks(xticks,fontsize=24)
plt.axhline(y =0.3, color = '#fefcff', linestyle = '-', lw=1.5)
plt.axvline(x=0.8, color = '#fefcff', linestyle = '-', lw=1.5)
plt.plot(0.8, 0.3, "o", color="black", ms=7,  mec='black', mew=0.5)

##Save the figure=================================================================================================
plt.savefig("Fig2_50th_round.png",dpi=600)
plt.show()


