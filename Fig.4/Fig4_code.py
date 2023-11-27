import numpy as np
import matplotlib
from numba import jit,njit,prange

## Prisoner's Dilemma=============================================================================================================
r=2.0
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

## Marginal liklihood=============================================================================================================

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

## Action action for first iteration============================================================================================

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

## Find indexing of the maximum element=============================================================================================

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


#=================================================================================================================================

## Payoff of when two reactive players play the game===============================================================================

#=================================================================================================================================

@njit
def Reactive_Reactive(Reactive_strategy_1,Reactive_strategy_2, average,no_of_round):

    #Initialization of total accumulated payoff=============
    total_payoff_reactive_1=0


    for e in range(average):

        ##Intialization of payoff for one realization==================

        payoff_rective_1=0

        #Starting the game =================================================

        for k in range(0,no_of_round):

            ##Choosing action====================================

            if k==0:
                action_reactive_1=First_iteration_action(Reactive_strategy_1)
                action_reactive_2=First_iteration_action(Reactive_strategy_2)  
            else:
                prev_action_1=action_reactive_1
                prev_action_2=action_reactive_2
                action_reactive_1=Action(Reactive_strategy_1,prev_action_2)
                action_reactive_2=Action(Reactive_strategy_2,prev_action_1)

            ## Receiving payoff =================================

            payoff_rective_1+=(discount_factor**k)*payoff(action_reactive_1, action_reactive_2)

        ##Accumulating payoff from each realization=================================

        total_payoff_reactive_1+=payoff_rective_1

    return total_payoff_reactive_1/average


#====================================================================================================================================

#One reactive player and one bayesian player ========================================================================================

#====================================================================================================================================


@njit      
def Reactive_Bayesian(Reactive_strategy,Bayesian_strategy,average,no_of_round):

    #Initialization of total accumulated payoff=============

    total_payoff_reactive=0
    total_payoff_Bayesian=0

    for e in range(average):
        ##Coping the strategy and use it for each realization from scratch============

        reactive_strategy=Reactive_strategy.copy()
        bayesian_strategy=Bayesian_strategy.copy()
        
 
        ##Intialization of payoff for one realization==================

        payoff_reactive=0
        payoff_Bayesian=0

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

            ## Receiving payoff =================================

            payoff_reactive+=(discount_factor**k)*payoff(action_reactive,action_bayesian)
            payoff_Bayesian+=(discount_factor**k)*payoff(action_bayesian,action_reactive)            

            ##Calculating cooperation level based on true strategy of player==============

            reactive_strategy[2]=Reactive_strategy[0]*cop_Bayesian[k]+ Reactive_strategy[1]*(1-cop_Bayesian[k])
            bayesian_strategy[2]=bayesian_strategy[0]*cop_Reactive[k]+bayesian_strategy[1]*(1-cop_Reactive[k])

        ##Accumulating payoff from each realization=================================
    
        total_payoff_reactive+=payoff_reactive
        total_payoff_Bayesian+=payoff_Bayesian

    return total_payoff_reactive/average, total_payoff_Bayesian/average

#================================================================================================================================

#Payoff of two Bayesian players interaction===================================================================================================

#================================================================================================================================


@njit
def Bayesian_Bayesian(Bayesian_strategy_1,Bayesian_strategy_2,average,no_of_round):

    #Initialization of total accumalated payoff=============

    total_payoff_Bayesian_1=0.0
    total_payoff_Bayesian_2=0.0

    for e in range(average):

        ## Coping the strategy and use it for each realization from scratch============

        bayesian_strategy_1=Bayesian_strategy_1.copy()
        bayesian_strategy_2=Bayesian_strategy_2.copy()

        ##Intialization of payoff for one realization========================

        payoff_Bayesian_1=0.0
        payoff_Bayesian_2=0.0

        ## Initilazation of cooperation level array ===========================

        cop_Bayesian_1=[]
        cop_Bayesian_2=[]

        ## Initialization of prior distribution which is taken as uniform, (N,N) is the shape of the distribution ===============

        prior_distribution_bayesian_1=prior_distribution_bayesian_2=np.ones((N,N))

        ## Starting the game=====================================================

        for k in range(0,no_of_round):

            ## Store the cooperation level at each iteration=====================

            cop_Bayesian_1.append(bayesian_strategy_1[2])
            cop_Bayesian_2.append(bayesian_strategy_2[2])

            ##Choosing action====================================

            if k==0:
                action_bayesian_1=First_iteration_action(bayesian_strategy_1)
                action_bayesian_2=First_iteration_action(bayesian_strategy_2)  

                ## Updating prior=====================================
    

                prior_distribution_bayesian_1,bayesian_strategy_1[0],bayesian_strategy_1[1]=Bayes_update(action_bayesian_2, p, q, bayesian_strategy_1[2], prior_distribution_bayesian_1)
                prior_distribution_bayesian_2,bayesian_strategy_2[0],bayesian_strategy_2[1]=Bayes_update(action_bayesian_1, p, q, bayesian_strategy_2[2], prior_distribution_bayesian_2)
            else:
                prev_action_bayesian_1=action_bayesian_1
                prev_action_bayesian_2=action_bayesian_2
                action_bayesian_1=Action(bayesian_strategy_1,prev_action_bayesian_2)
                action_bayesian_2=Action(bayesian_strategy_2,prev_action_bayesian_1)

                ## Updating prior=====================================

                prior_distribution_bayesian_1,bayesian_strategy_1[0],bayesian_strategy_1[1]=Bayes_update(action_bayesian_2, p, q, cop_Bayesian_1[k-1], prior_distribution_bayesian_1)
                prior_distribution_bayesian_2,bayesian_strategy_2[0],bayesian_strategy_2[1]=Bayes_update(action_bayesian_1, p, q, cop_Bayesian_2[k-1], prior_distribution_bayesian_2)
            
            ## Receiving payoff =================================

            payoff_Bayesian_1+=(discount_factor**k)*payoff(action_bayesian_1,action_bayesian_2)
            payoff_Bayesian_2+=(discount_factor**k)*payoff(action_bayesian_2,action_bayesian_1)

            
            ##Calculating cooperation level based on true strategy of player==============

            bayesian_strategy_2[2]=bayesian_strategy_2[0]*cop_Bayesian_1[k]+bayesian_strategy_2[1]*(1-cop_Bayesian_1[k])
            bayesian_strategy_1[2]=bayesian_strategy_1[0]*cop_Bayesian_2[k]+bayesian_strategy_1[1]*(1-cop_Bayesian_2[k])
        
        ##Accumulating payoff from each realization=================================

        total_payoff_Bayesian_1+=payoff_Bayesian_1
        total_payoff_Bayesian_2+=payoff_Bayesian_2

    return (total_payoff_Bayesian_1+total_payoff_Bayesian_2)/(2*average)


## Parameters=======================================================================================================================

no_of_round=700
average=10000
discount_factor=0.75
N=11
p=q=np.linspace(0,1,N)


## ESS in p-q parameter space, generating the data=================================================================================

value_p=np.linspace(0,1,51)
value_q=np.linspace(0,1,51)

@njit(parallel=True)
def abcd(value_p,value_q):
    a_matrix=np.zeros((len(value_p),len(value_q)))
    b_matrix=np.zeros((len(value_p),len(value_q)))
    c_matrix=np.zeros((len(value_p),len(value_q)))
    d_matrix=np.zeros((len(value_p),len(value_q)))
    for i in prange(len(value_q)):
        print(i)
        for j in prange(len(value_p)):
                Bayesian_strategy=np.array([value_p[j],value_q[i],0.5])
                Reactive_strategy=np.array([value_p[j],value_q[i],0.5])
                a=Reactive_Reactive(Reactive_strategy,Reactive_strategy, average, no_of_round)
                b,c=Reactive_Bayesian(Reactive_strategy, Bayesian_strategy, average, no_of_round)
                d=Bayesian_Bayesian(Bayesian_strategy,Bayesian_strategy, average, no_of_round)
                a_matrix[i][j]=a
                b_matrix[i][j]=b
                c_matrix[i][j]=c
                d_matrix[i][j]=d
    return a_matrix,b_matrix,c_matrix,d_matrix
a,b,c,d=abcd(value_p,value_q)


#### Save the data ===============================================================================================================

np.savetxt('matrix_element_aRR_df=0.75_c=0.5_r=2.0_avg_10k_pq_51.txt',a)
np.savetxt('matrix_element_bRB_df=0.75_c=0.5_r=2.0_avg_10k_pq_51.txt',b)
np.savetxt('matrix_element_cBR_df=0.75_c=0.5_r=2.0_avg_10k_pq_51.txt',c)
np.savetxt('matrix_element_dBB_df=0.75_c=0.5_r=2.0_avg_10k_pq_51.txt',d)
