import math
import matplotlib.pyplot as plt
from numpy import *

	#Drop based Learning rate
def step_decay(epoch,initial_lrate,lrate_drop,lrate_epochs_drop):
	initial_lrate = initial_lrate
	drop = lrate_drop
	epochs_drop = lrate_epochs_drop
	lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
	return lrate

epochs = 20
initial_lrate = [0.1,0.06,0.01,0.001]
lrate_drop = 0.5
lrate_epochs_drop = 5
lrate = {}
lrate['0.01'] = []
lrate['0.1'] = []
lrate['0.001'] = []
lrate['0.06'] = []
for l in range(0,4):
	for epoch in range(0,epochs+1):
		lrate[str(initial_lrate[l])].append(step_decay(epoch,initial_lrate[l],lrate_drop,lrate_epochs_drop))

x_ticks = [1,4,8,12,16,20]
y_ticks = [0.0,0.0005,0.001,0.002,0.004,0.006,0.008,0.009,0.01]
x_tick_labels = ['1','4','8','12','16','20']
#y_tick_labels = ['0.000','0.01','0.02','0.04','0.06','0.08','0.1','0.12']
y_tick_labels = ['0.0','0.0005','0.001','0.002','0.004','0.006','0.008','0.009','0.01']
fig = plt.figure(figsize=(25,15))
#plt.plot(lrate['0.1'],'r',linewidth=3)
#plt.plot(lrate['0.06'],'g',linewidth=3)
plt.plot(lrate['0.01'],'b',linewidth=3)
#plt.plot(lrate['0.001'],linewidth=3)

plt.ylabel('Learning Rate', fontsize=35)
plt.xlabel('Epochs', fontsize=35)
plt.xticks(x_ticks, x_tick_labels, fontsize=35)
plt.yticks(y_ticks, y_tick_labels, fontsize=35)
fig.tight_layout()
plt.savefig('lr_scheduler.eps', format='eps', dpi=6000)
plt.show()
