import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator


rangenet_runtime = []
rangenet_miou = []

rangenet_runtime.append(1000/7)
rangenet_miou.append(49.9)

rangenet_runtime.append(1000/11)
rangenet_miou.append(45.4)

rangenet_runtime.append(1000/22)
rangenet_miou.append(39.3)




rangenet2_runtime = []
rangenet2_miou = []

rangenet2_runtime.append(1000/5)
rangenet2_miou.append(52.2)

rangenet2_runtime.append(1000/8)
rangenet2_miou.append(48.0)

rangenet2_runtime.append(1000/13)
rangenet2_miou.append(41.9)

# *****************************************

minet_runtime = []
minet_miou = []

minet_runtime.append(1000/24)
minet_miou.append(52.4)

minet_runtime.append(1000/47)
minet_miou.append(49.1)

minet_runtime.append(1000/80)
minet_miou.append(45.0)

minet2_runtime = []
minet2_miou = []

minet2_runtime.append(1000/13)
minet2_miou.append(55.2)

minet2_runtime.append(1000/18)
minet2_miou.append(52.4)

minet2_runtime.append(1000/21)
minet2_miou.append(48.5)


f,ax = plt.subplots(1,1,sharey=True, facecolor='w')

# plot the same data on both axes
l1 = ax.plot(rangenet_runtime, rangenet_miou, marker='o', color='b', alpha=0.5, linestyle='-.')
l2 = ax.plot(rangenet2_runtime, rangenet2_miou, marker='o', color='b', alpha=1.0, linestyle='-.')
l3 = ax.plot(minet_runtime, minet_miou, marker='s', color='r', alpha=0.5, linestyle='-.')
l4 = ax.plot(minet2_runtime, minet2_miou, marker='s', color='r', alpha=1.0, linestyle='-.')

ax.set_xlim(0,230)
ax.set_ylim(30,60)


# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright='off')

locator_y = MultipleLocator(10)
locator = MultipleLocator(50)
ax.xaxis.set_major_locator(locator)
ax.yaxis.set_major_locator(locator_y)
locator2 = MultipleLocator(2000)


ax.set_ylabel('Mean IoU [%]') # y value
ax.set_xlabel('Scan/ms') # y value
# # plt.grid(True)
plt.legend(['RangeNet53', 'RangeNet53++', 'MINet', 'MINet++'], loc=4)
plt.axvline(x=100, color='r', linestyle=':')

plt.show()