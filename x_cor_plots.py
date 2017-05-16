import matplotlib.pyplot as plt
import numpy as np
from os import listdir
import scipy.io as sio
names = listdir('output_files')



for n,name in enumerate(names[0:3]):
    mat_contents = sio.loadmat('output_files/' + name)
    xcors = mat_contents['xcors']
    if n==0:
        t_len = np.shape(xcors)[1]
        X_Y=np.zeros((3, t_len))
        Y_Z = np.zeros((3, t_len))
    X_Y[n,:]=xcors[0,:]
    Y_Z[n, :] = xcors[1, :]

t_vect=np.arange(-(t_len-1)/2*3,(t_len)/2*3,step=3)
fig,(ax,ax1)=plt.subplots(1,2,figsize=(10,4))
ax.errorbar(t_vect[0::6],np.mean(X_Y,0)[0::6],yerr=np.std(X_Y,0)[0::6],color='teal',fmt='o',alpha=0.75,label='WT')
ax1.errorbar(t_vect[0::6],np.mean(Y_Z,0)[0::6],yerr=np.std(Y_Z,0)[0::6],color='teal',fmt='o',alpha=0.75,label='WT')

for n,name in enumerate(names[3:6]):
    print(name)
    mat_contents = sio.loadmat('output_files/' + name)
    xcors = mat_contents['xcors']
    if n==0:
        t_len = np.shape(xcors)[1]
        X_Y=np.zeros((3, t_len))
        Y_Z = np.zeros((3, t_len))
    X_Y[n,:]=xcors[0,:]
    Y_Z[n, :] = xcors[1, :]

t_vect=np.arange(-(t_len-1)/2*3,(t_len)/2*3,step=3)

# ax.errorbar(t_vect[0::6],np.mean(X_Y,0)[0::6],yerr=np.std(X_Y,0)[0::6],color='orange',fmt='o',alpha=0.75,label='MV')
# ax1.errorbar(t_vect[0::6],np.mean(Y_Z,0)[0::6],yerr=np.std(Y_Z,0)[0::6],color='orange',fmt='o',alpha=0.75,label='MV')




ax1.set_ylim([-.2,1.2])
ax.set_ylim([-.2,1.2])
ax.set_xlabel('Tau')
ax1.set_xlabel('Tau')
ax.set_title('X*Y Cross-Correlation')
ax1.set_title('Y*Z Cross-Correlation')
ax1.legend()
fig.savefig('figures/xcor_1.png',bbox_inches='tight')