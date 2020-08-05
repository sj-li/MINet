import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

runtime = []
miou = []
size = []

# #pointnet
# runtime.append(1000/2)
# miou.append(14.6)
# size.append(3)

# #pointnet2
# runtime.append(1000/0.1)
# miou.append(20.1)
# size.append(6)

# #SPGraph
# runtime.append(1000/0.2)
# miou.append(20)
# size.append(0.25)

# #SPLATNet
# runtime.append(1000/1)
# miou.append(22.8)
# size.append(0.8)

# #TangentConv
# runtime.append(1000/0.3)
# miou.append(35.9)
# size.append(0.4)

#PolarSeg
runtime.append(1000/8)
miou.append(54.3)
size.append(0.4)

#RandLA
runtime.append(1000/16)
miou.append(53.9)
size.append(0.4)


# *****************************************

image_runtime = []
image_miou = []
image_size = []

#deeplab
image_runtime.append(1000/39)
image_miou.append(38.6)
image_size.append(3)

#bisenet
image_runtime.append(1000/50)
image_miou.append(37.1)
image_size.append(6)

#denseaspp
image_runtime.append(1000/20)
image_miou.append(40.2)
image_size.append(0.25)

#pspnet
image_runtime.append(1000/31)
image_miou.append(44.4)
image_size.append(0.8)

# *************************************************

projected_runtime = []
projected_miou = []
projected_size = []
projected_text = []

#squeezeseg
projected_runtime.append(1000/66)
projected_miou.append(29.5)
projected_size.append(1)
projected_text.append('SqueezeSeg (1M)')

#squeezeseg_crf
projected_runtime.append(1000/55)
projected_miou.append(30.8)
projected_size.append(1)
projected_text.append('SqueezeSeg_CRF (1M)')

#squeezeseg_v2
projected_runtime.append(1000/50)
projected_miou.append(39.7)
projected_size.append(1)
projected_text.append('SqueezeSegV2 (1M)')

# squeezeseg_v2_crf
projected_runtime.append(1000/40)
projected_miou.append(39.6)
projected_size.append(1)
projected_text.append('SqueezeSegV2_CRF (1M)')

#rangenet21
projected_runtime.append(1000/20)
projected_miou.append(47.4)
projected_size.append(25)
projected_text.append('RangeNet21 (25M)')

#rangenet53
projected_runtime.append(1000/13)
projected_miou.append(49.9)
projected_size.append(50)
projected_text.append('RangeNet53 (50M)')

#rangenet53++
projected_runtime.append(1000/12)
projected_miou.append(52.2)
projected_size.append(50)
projected_text.append('RangeNet53++ (50M)')

#minet
projected_runtime.append(1000/59)
projected_miou.append(52.4)
projected_size.append(50)
projected_text.append('RangeNet53 (50M)')

#minet++
projected_runtime.append(1000/47)
projected_miou.append(55.2)
projected_size.append(50)
projected_text.append('RangeNet53++ (50M)')

size = np.array(size)*50

f,ax = plt.subplots(1,1,sharey=True, facecolor='w')

# plot the same data on both axes
l1 = ax.scatter(projected_runtime, projected_miou, s=40, marker='o', color='b', alpha=0.8, label='Projected-based')
l2 = ax.scatter(image_runtime, image_miou, s=40, marker='*', color='g', alpha=0.8, label='Image-based')
l3 = ax.scatter(runtime, miou, s=40, marker='s', color='r', alpha=0.8, label='Point-based')

ax.set_xlim(0,150)
ax.set_ylim(25,60)


# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright='off')

# locator_y = MultipleLocator(10)
# locator = MultipleLocator(20)
# ax.xaxis.set_major_locator(locator)
# ax.yaxis.set_major_locator(locator_y)
# locator2 = MultipleLocator(2000)

plt.yticks([25, 35, 45, 55])

ax.set_ylabel('Mean IoU [%]') # y value
ax.set_xlabel('Scan/ms')
# plt.grid(True)
plt.legend([l1, l2, l3], ['Projected-based', 'Image-based', 'Point-based'])

plt.show()