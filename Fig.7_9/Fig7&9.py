import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import rcParams

## Setting style of font and size of the plot==================================================================================================================
rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]=20
rcParams['axes.linewidth'] =2.0
rcParams['figure.figsize'] = [8,8]
rcParams['xtick.major.size'] = 9
rcParams['xtick.major.width'] = 2.0
rcParams['xtick.minor.size'] = 4
rcParams['xtick.minor.width'] = 2.0
rcParams['ytick.major.size'] = 9
rcParams['ytick.major.width'] = 2.0
rcParams['ytick.minor.size'] = 4


## Import the data, we draw the figures in Fig.7 and Fig.9 using the same data used for Fig.4, Fig.5 & Fig.6.================================================== 

Payoff_matrix_element_a= np.loadtxt('matrix_element_aRR_df=0.99_c=0.5_r=10.0_avg_10k_pq_51.txt') ## (a) Payoff of reactive player for reactive-rective interaction
Payoff_matrix_element_b= np.loadtxt('matrix_element_bRB_df=0.99_c=0.5_r=10.0_avg_10k_pq_51.txt') ## (b) Payoff of reactive player for reactive-Bayesian interaction
Payoff_matrix_element_c= np.loadtxt('matrix_element_cBR_df=0.99_c=0.5_r=10.0_avg_10k_pq_51.txt') ## (c) Payoff of Bayesian player for reactive-Bayesian interaction
Payoff_matrix_element_d= np.loadtxt('matrix_element_dBB_df=0.99_c=0.5_r=10.0_avg_10k_pq_51.txt') ## (d) Payoff of Bayesian player for Bayesian-Bayesian interaction



## Module to find the ESS for a finte population=============================================================================================
def ESSN(a,b,c,d,N):
	if b*(N-1) <c+d*(N-2) and a*(N-2) + b*(2*N-1) <c*(N+1) +d*(2*N-4) and c*(N-1) <b+a*(N-2) and d*(N-2) + c*(2*N-1) <b*(N+1) +a*(2*N-4):
		return 40 # Both (Reactive and Bayesian) are ESSN
	elif b*(N-1) <c+d*(N-2) and a*(N-2) + b*(2*N-1) <c*(N+1) +d*(2*N-4):
		return 30 ## Bayesian strategy is ESSN
	elif c*(N-1) <b+a*(N-2) and d*(N-2) + c*(2*N-1) <b*(N+1) +a*(2*N-4):
		return 20 ## Reactive strategy is ESSN
	else:
		return 10 ## No ESS
##==========================================================================================================================================

N=2 ##Population Size


value_p=value_q=np.linspace(0,1,51)
ESS_N_matrix=np.zeros((len(value_p), len(value_q)))
for i in range(len(value_p)):
	for j in range(len(value_p)):
		a=round(Payoff_matrix_element_a[i][j],2)
		b=round(Payoff_matrix_element_b[i][j],2)
		c=round(Payoff_matrix_element_c[i][j],2)
		d=round(Payoff_matrix_element_d[i][j],2)
		ESS_N_matrix[i][j]=ESSN(a,b,c,d,N)

##==========================================================================================================================================

col_dict={
          10:"#ffffff",
          20:"#df0100",
          30:"#004c00",
          40:"#0000b5"}

cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
labels = np.array(["NO ESS","Reactive","Bayesian", "B & R"])
len_lab = len(labels)
norm_bins = np.sort([*col_dict.keys()]) + 0.5
norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)

## Make normalizer and formatter==========================================================
norm = matplotlib.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
fmt = matplotlib.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])

# Plot our figure=========================================================================
fig,ax = plt.subplots()
im = ax.pcolor(value_q,value_p,ESS_N_matrix, cmap=cm, norm=norm,shading='auto' )
diff = norm_bins[1:] - norm_bins[:-1]
tickz = norm_bins[:-1] + diff / 2
xticks=[0.0,0.5,1.0]
plt.xticks(xticks, fontsize=28)
plt.yticks(xticks,fontsize=28)
fig.savefig("Fig7(a).png",dpi=600)
plt.show()        
