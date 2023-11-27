import numpy as np
import matplotlib.pyplot as plt
from numba import jit,njit,prange
import time
from pylab import rcParams
start=time.time()

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]=16
rcParams['axes.linewidth'] =2.0

## Import the data======================================================================================
a=np.loadtxt('matrix_element_aRR_df=0.99_p=0.2_q=0.7_r=2.0_avg_10k_round_700.txt')
b=np.loadtxt('matrix_element_bRB_df=0.99_p=0.2_q=0.7_r=2.0_avg_10k_round_700.txt')
c=np.loadtxt('matrix_element_cBR_df=0.99_p=0.2_q=0.7_r=2.0_avg_10k_round_700.txt')
d=np.loadtxt('matrix_element_dBB_df=0.99_p=0.2_q=0.7_r=2.0_avg_10k_round_700.txt')

##======================================================================================================
no_of_round=len(a)
rounds=np.arange(1,no_of_round+1,1)
xticks=[1,175,350,525,700]
yticks=[0,17,34,51,68,85]

## Draw the plot======================================================================================
plt.plot(rounds,a, color='#ff0000', linestyle='solid', lw=2.3)
plt.plot(rounds,b,color='#017500', linestyle='dashed', lw=2.5)
plt.plot(rounds,c, color='#fe0dff',linestyle='dotted', lw=3.0)
plt.plot(rounds,d, color='#0000ff',linestyle='dashdot', lw=2.5)
plt.xlim(1,700)
plt.xticks(xticks, fontsize=28)
plt.yticks(yticks,fontsize=28)
plt.savefig("Fig3(c).png",dpi=600)
plt.show()