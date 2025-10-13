# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 17:35:37 2021
作图用的，x轴为水文水动力学模拟结果，y轴为深度学习模拟结果
@author: dell
"""
from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
import numpy as np
import pandas

table_path="E:/Wenjie/deepLearning_floodMapping/maxDepMapping2/数据分析/20210715结果对比/test.asc"
dataframe = pandas.read_table(table_path,header=None,sep=' ')
dataset = dataframe.values

x=np.empty([36445], dtype = float)
y=np.empty([36445], dtype = float)
for k in range(36445):
    x[k]=dataset[k,0].astype('float')
    y[k]=dataset[k,1].astype('float')


plt.hist2d(x, y, bins=100, norm=LogNorm())
plt.colorbar()
plt.title('Rain 17')
plt.xlabel('HM')
plt.ylabel('DLM')
plt.show()

# from scipy.stats import gaussian_kde

# # Generate fake data


# # Calculate the point density
# xy = np.vstack([x,y])
# z = gaussian_kde(xy)(xy)

# # Sort the points by density, so that the densest points are plotted last
# idx = z.argsort()
# x, y, z = x[idx], y[idx], z[idx]

# fig, ax = plt.subplots()
# plt.scatter(x, y,c=z,  s=20,cmap='Spectral')
# plt.colorbar()
# plt.show()