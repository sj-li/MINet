import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.pyplot import MultipleLocator

plt.rcParams.update({'font.size': 12})

# plt.title('title name')  # Title  
plt.xlabel('Distance to sensor [m]') # x value
plt.ylabel('Mean IoU [%]') # y value
plt.grid(True)

dist = [10, 20, 30, 40, 50]

squeezeseg = [31.3, 29.7, 24.8, 19.9, 16.2]
squeezeseg_crf = [32.6, 31.6, 25.3, 18.4, 14.2]

squeezesegV2 = [44.1, 38.6, 30.3, 24.1, 19.9]
squeezesegV2_CRF = [44.2, 38.5, 29.7, 22.7, 18.2]

rangenet21_knn = [52.8, 49.5, 40.5, 33.7, 28.8]

rangenet53_2048_knn = [56.5, 50.6, 40.3, 34.4, 29.3]
rangenet53_1024_knn = [37.8, 38.2, 31.1, 26.0, 22.0]
rangenet53_512_knn = [44.6, 37.7, 27.7, 21.8, 18.2]

minet_2048_knn = [59.5, 53.6, 41.3, 34.0, 29.3]
minet_2048 = [57.5, 49.9, 38.6, 31.6, 27.1]
# minet_2048_knn = [57.7, 52.7, 41.5, 34.4, 29.3]
# minet_2048 = [56, 49.1, 38.8, 32.0, 27.3]

minet_1024_knn = [57.0, 50.1, 38.2, 31.1, 26.6]
minet_1024 = [54.6, 46.1, 35.3, 28.6, 24.4]

minet_512_knn = [53.5, 45.5, 33.7, 27.2, 23.4]
minet_512 = [50.9, 41.2, 30.8, 25.0, 21.4]

# **************************************************************************************************************************

# plt.plot(dist, minet_2048_knn, color='r', linestyle='-', marker='o', label='MINet++-2048')
# plt.plot(dist, minet_1024_knn, color='r', linestyle='--', marker='o', label='MINet++-1024')
# plt.plot(dist, minet_512_knn, color='r', linestyle=':', marker='o', label='MINet++-512')

# plt.plot(dist, rangenet53_2048_knn, color='b', linestyle='-', marker='s', label='RangeNet53++-2048')
# plt.plot(dist, rangenet53_1024_knn, color='b', linestyle='--', marker='s', label='RangeNet53++-1024')
# ax = plt.plot(dist, rangenet53_512_knn, color='b', linestyle=':', marker='s', label='RangeNet53++-512')

# plt.xticks([10, 20, 30, 40, 50])
# plt.yticks([10, 20, 30, 40, 50])

# plt.legend(['MINet++-2048', 'MINet++-1024', 'MINet++-512', 'RangeNet53++-2048', 'RangeNet53++-1024', 'RangeNet53++-512', 'RangeNet21', 'SqueezeSeg_CRF', 'SqueezeSegV2_CRF'], fontsize='small')

# **************************************************************************************************************************

# **************************************************************************************************************************

plt.plot(dist, minet_2048_knn, color='r', linestyle='-', marker='o', label='MINet++')
plt.plot(dist, rangenet53_2048_knn, color='b', linestyle='-', marker='s', label='RangeNet53++')
plt.plot(dist, rangenet21_knn, color='g', linestyle='-', marker='^', label='RangeNet21++')
plt.plot(dist, squeezeseg_crf, color='c', linestyle='-', marker='*', label='SqueezeSeg_CRF')
plt.plot(dist, squeezesegV2_CRF, color='y', linestyle='-', marker='.', label='SqueezeSegV2_CRF')

plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(10))

plt.legend(['MINet++', 'RangeNet53++', 'RangeNet21++', 'SqueezeSeg-CRF', 'SqueezeSegV2-CRF'])

# **************************************************************************************************************************

plt.show()

# plt.xlabel('Distance to sensor [m]') # x value
# plt.ylabel('Accuracy [%]') # y value
# plt.grid(True)

# minet_2048_knn = [89.6, 88.3, 82.5, 77.8, 76.6]
# minet_2048 = [88.9, 86.5, 80.2, 75.4, 73.9]

# minet_1024_knn = [89.7, 87.6, 81.4, 75.6, 75.3]
# minet_1024 = [88.8, 85.3, 78.6, 72.7, 71.7]

# minet_512_knn = [89.4, 87.2, 79.6, 75.9, 74.4]
# minet_512 = [88.3, 84.3, 76.1, 71.6, 69.4]

# squeezeseg = 
# squeezeseg_crf = 

# squeezesegV2 = 
# squeezesegV2_CRF = 

# rangenet21_knn = []

# rangenet53_2048_knn = [90.2, 89.3, 84.5, 80.2, 78.4]
# rangenet53_1024_knn = [84.2, 85.4, 79.5, 74.4, 71.1]
# rangenet53_512_knn = [86.9, 84.3, 75.4, 66.9, 62.8]

# plt.plot(dist, minet_2048_knn, color='r', linestyle='-', marker='o', label='MINet++-2048')
# plt.plot(dist, minet_1024_knn, color='r', linestyle='--', marker='o', label='MINet++-1024')
# plt.plot(dist, minet_512_knn, color='r', linestyle=':', marker='o', label='MINet++-512')

# plt.plot(dist, rangenet53_2048_knn, color='b', linestyle='-', marker='s', label='RangeNet53++-2048')
# plt.plot(dist, rangenet53_1024_knn, color='b', linestyle='--', marker='s', label='RangeNet53++-1024')
# plt.plot(dist, rangenet53_512_knn, color='b', linestyle=':', marker='s', label='RangeNet53++-512')

# plt.legend(['MINet++-2048', 'MINet++-1024', 'MINet++-512', 'RangeNet53++-2048', 'RangeNet53++-1024', 'RangeNet53++-512'])

# plt.show()