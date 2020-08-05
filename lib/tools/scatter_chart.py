import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Fixing random state for reproducibility
# np.random.seed(19680801)


# N = 50
# x = np.random.rand(N)
# y = np.random.rand(N)
# colors = np.random.rand(N)
# area = (30 * np.random.rand(N))**2  # 0 to 15 point radii

# plt.scatter(x, y, s=area, c=colors, alpha=0.5)
# plt.show()

runtime = []
miou = []
size = []

#pointnet
runtime.append(1000/2)
miou.append(14.6)
size.append(3)

#pointnet
runtime.append(1000/0.1)
miou.append(20.1)
size.append(6)

#SPGraph
runtime.append(1000/0.2)
miou.append(20)
size.append(0.25)

#SPLATNet
runtime.append(1000/1)
miou.append(22.8)
size.append(0.8)

#TangentConv
runtime.append(1000/0.3)
miou.append(35.9)
size.append(0.4)

# #PolarSeg
# runtime.append(1000/8)
# miou.append(54.3)
# size.append(0.4)

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
projected_miou.append(51.4)
projected_size.append(50)
projected_text.append('RangeNet53 (50M)')

#minet++
projected_runtime.append(1000/47)
projected_miou.append(54.1)
projected_size.append(50)
projected_text.append('RangeNet53++ (50M)')

size = np.array(size)*50

f,(ax,ax2) = plt.subplots(1,2,sharey=True, facecolor='w')

# plot the same data on both axes
l1 = ax.scatter(projected_runtime, projected_miou, s=20, marker='o', color='b', alpha=0.8, label='Projected-based')
l2 = ax.scatter(image_runtime, image_miou, s=20, marker='*', color='g', alpha=0.8, label='Image-based')
l4 = ax.scatter(image_runtime, image_miou, s=20, marker='*', color='g', alpha=0.8, label='Image-based')
l3 = ax2.scatter(runtime, miou, s=20, marker='s', color='r', alpha=0.8, label='Point-based')

ax.set_xlim(0,80)
ax2.set_xlim(0,10500)
ax.set_ylim(0,60)
ax2.set_ylim(0,60)


# hide the spines between ax and ax2
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax.yaxis.tick_left()
ax.tick_params(labelright='off')
ax2.yaxis.tick_right()
ax2.tick_params(labelright='off')

locator_y = MultipleLocator(10)
locator = MultipleLocator(20)
ax.xaxis.set_major_locator(locator)
ax.yaxis.set_major_locator(locator_y)
locator2 = MultipleLocator(2000)
ax2.xaxis.set_major_locator(locator2)
ax2.yaxis.set_major_locator(locator_y)

# This looks pretty good, and was fairly painless, but you can get that
# cut-out diagonal lines look with just a bit more work. The important
# thing to know here is that in axes coordinates, which are always
# between 0-1, spine endpoints are at these locations (0,0), (0,1),
# (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# appropriate corners of each of our axes, and so long as we use the
# right transform and disable clipping.

plt.subplots_adjust(wspace=0.05)
d = .015 # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((1-d,1+d), (-d,+d), **kwargs)
# ax.plot((1-d,1+d),(1-d,1+d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
# ax2.plot((-d,+d), (1-d,1+d), **kwargs)
ax2.plot((-d,+d), (-d,+d), **kwargs)

# What's cool about this is that now if we vary the distance between
# ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# the diagonal lines will move accordingly, and stay right at the tips
# of the spines they are 'breaking'

ax.set_ylabel('Mean IoU [%]') # y value
ax2.set_xlabel('Scan/ms') # x value
# plt.grid(True)
plt.legend([l1, l2, l3], ['Point-based', 'Image-based', 'Projected-based'])

plt.show()

# f, (lax, rax) = plt.subplots(1, 2, sharex=True)

# lax.scatter(projected_runtime, projected_miou, s=100, marker='o', color='b', alpha=0.8, label='Projected-based')
# rax.scatter(runtime, miou, s=100, marker='s', color='r', alpha=0.8, label='Point-based')

# lax.set_xlim(450, 5500)  # outliers only
# rax.set_xlim(0, 90)  # most of the data

# # hide the spines between ax and ax2
# lax.spines['right'].set_visible(False)
# rax.spines['left'].set_visible(False)
# lax.yaxis.tick_left()
# lax.tick_params(labelright=False)  # don't put tick labels at the top
# rax.yaxis.tick_right()


# # This looks pretty good, and was fairly painless, but you can get that
# # cut-out diagonal lines look with just a bit more work. The important
# # thing to know here is that in axes coordinates, which are always
# # between 0-1, spine endpoints are at these locations (0,0), (0,1),
# # (1,0), and (1,1).  Thus, we just need to put the diagonals in the
# # appropriate corners of each of our axes, and so long as we use the
# # right transform and disable clipping.

# d = .015  # how big to make the diagonal lines in axes coordinates
# # arguments to pass to plot, just so we don't keep repeating them
# kwargs = dict(transform=lax.transAxes, color='k', clip_on=False)
# lax.plot((1-d, 1+d), (-d, +d), **kwargs)        # top-left diagonal
# lax.plot((1-d, 1+d), (1-d, 1+d), **kwargs)  # top-right diagonal

# kwargs.update(transform=rax.transAxes)  # switch to the bottom axes
# rax.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
# rax.plot((-d, +d), (-d, +d), **kwargs)  # bottom-right diagonal

# # What's cool about this is that now if we vary the distance between
# # ax and ax2 via f.subplots_adjust(hspace=...) or plt.subplot_tool(),
# # the diagonal lines will move accordingly, and stay right at the tips
# # of the spines they are 'breaking'

# plt.show()


# plt.scatter(runtime, miou, s=100, marker='s', color='r', alpha=0.8, label='Point-based')
# plt.scatter(projected_runtime, projected_miou, s=100, marker='o', color='b', alpha=0.8, label='Projected-based')



# plt.xlabel('Scan/ms') # x value
# plt.ylabel('Mean IoU [%]') # y value
# # plt.grid(True)
# plt.legend(['Point-based', 'Projected-based'])

# plt.show()

# # zip joins x and y coordinates in pairs
# for x,y, t in zip(projected_runtime,projected_miou, projected_text):

#     label = "{:.2f}".format(y)

#     # this method is called for each point
#     plt.annotate(t, # this is the text
#                  (x+800,y-2), # this is the point to label
#                  textcoords="offset points", # how to position the text
#                  xytext=(0,10), # distance from text to points (x,y)
#                  ha='center') # horizontal alignment can be left, right or center