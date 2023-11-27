import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from pylab import rcParams
import matplotlib

rcParams["font.family"] = "Times New Roman"
rcParams["font.size"]=16
rcParams['axes.linewidth'] =1.5
rcParams['xtick.major.size'] = 7
rcParams['xtick.major.width'] = 1.5
rcParams['xtick.minor.size'] = 3
rcParams['xtick.minor.width'] = 1.5
rcParams['ytick.major.size'] = 7
rcParams['ytick.major.width'] = 1.5
rcParams['ytick.minor.size'] = 3
rcParams['ytick.minor.width'] = 1.5

## Import the data==========================================================================================
a=np.loadtxt('matrix_element_aRR_df=0.75_r=10.0_gamma=0.3_c=0.5_avg_10k_pq_51.txt')
b=np.loadtxt('matrix_element_bRB_df=0.75_r=10.0_gamma=0.3_c=0.5_avg_10k_pq_51.txt')
c=np.loadtxt('matrix_element_cBR_df=0.75_r=10.0_gamma=0.3_c=0.5_avg_10k_pq_51.txt')
d=np.loadtxt('matrix_element_dBB_df=0.75_r=10.0_gamma=0.3_c=0.5_avg_10k_pq_51.txt')


## Module to find the ESS for an infinite population========================================================
def ESS(a,b,c,d):
	if a>c and d>b:
		return 1.5  #Both (Reactive & Bayesian) are ESSs
	elif a>c or (a==c and b>d):
		return 1.0 #Reactive is ESS
	elif d>b or (d==b and c>a):
		return 0.0 # Bayesian is ESS
	elif c>a and b>d:
		return (b-d)/(b-d+c-a) # Mixed ESS
	else:
		return -0.5 #NO ESS

## Making the ESS matrix to plot============================================================================
p=q=np.linspace(0,1,51)
ESS_matrix=np.zeros((len(p),len(q)))
for i in range(len(a)):
	for j in range(len(a)):
		a[i][j]=round(a[i][j],2)
		b[i][j]=round(b[i][j],2)
		c[i][j]=round(c[i][j],2)
		d[i][j]=round(d[i][j],2)
		ESS_matrix[i][j]=ESS(a[i][j],b[i][j],c[i][j],d[i][j])


## Making the colorbar======================================================================================           
def hex_to_rgb(value):
    value = value.strip("#")
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    return [v/256 for v in value]


def get_continuous_cmap(hex_list, float_list=None):
    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0,1,len(rgb_list)))
        
    cdict = dict()
    for num, col in enumerate(['red', 'green', 'blue']):
        col_list = [[float_list[i], rgb_list[i][num], rgb_list[i][num]] for i in range(len(float_list))]
        cdict[col] = col_list
    cmp = LinearSegmentedColormap('my_cmp', segmentdata=cdict, N=256)
    return cmp

## Plot=====================================================================================================
viridis =get_continuous_cmap(["#ffffff", "#004c00", "#f4dc00", "#f4dc00","#df0100","#0000b5"])
xticks=[0.0,0.5,1.0] 
plt.imshow(ESS_matrix,vmin=-0.5, vmax=1.5, alpha=1.0, extent=[0, 1, 0, 1],cmap=viridis ,origin="lower")
plt.xticks(xticks, fontsize=30)
plt.yticks(xticks,fontsize=30)
plt.savefig("Fig6(c).png",dpi=600)
plt.show()
