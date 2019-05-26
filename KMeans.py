import pandas as pd
import numpy as np
from sklearn import cluster as sc
import matplotlib.pyplot as plt
import seaborn as sb

# Configure
DataIndex=20
VehicleCapacity=10
k=3
maxiter=10

# import data
RawData=pd.read_excel('Datas/VRP/VRP100/VRP100.xlsx')
DataSet=RawData.to_numpy()
DataSet=DataSet.reshape([1000,-1,3])
print('Data prepared with shape',DataSet.shape)

# data process
SelectedData=DataSet[DataIndex]
DepotCoordinate=SelectedData[-1][:2]
CustomersCoordinate=SelectedData[:-1][:,:2]
CustomersDemonds=SelectedData[:-1][:,2]
print('DepotCoordinate:',DepotCoordinate)
print('Total Capacity:',CustomersDemonds.sum())

# cluster
cludata=sc.k_means(CustomersCoordinate,k,max_iter=maxiter)
print('time cost:',cludata[2])

# data split
CustomData=[[] for i in range(k)]
for i,j in zip(CustomersCoordinate,cludata[1]):
    CustomData[j].append(i)

# visualization
fig,ax=plt.subplots()
sb.stripplot(x='X',y='Y',data=pd.DataFrame(CustomData[0],columns=['X','Y']),color='r')
sb.stripplot(x='X',y='Y',data=pd.DataFrame(CustomData[1],columns=['X','Y']),color='g')
sb.stripplot(x='X',y='Y',data=pd.DataFrame(CustomData[2],columns=['X','Y']),color='b')
plt.plot(DepotCoordinate[0],DepotCoordinate[1],marker='s')
plt.show()