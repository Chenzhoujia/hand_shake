# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 11:36:04 2016

@author: J. C. Vasquez-Correa
"""

import numpy as np
import math
from statsmodels.tsa.tsatools import lagmat
from sklearn.metrics.pairwise import euclidean_distances as dist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def Dim_Corr(datas, Tao, m, graph=False): 
	"""
	Compute the correlation dimension of a time series with a time-lag Tao and an embedding dimension m
	datas--> time series to compute the correlation dimension
	Tao--> time lag computed using the first zero crossing of the auto-correlation function (see Tao func)   
	m--> embeding dimension of the time-series, computed using the false neighbors method (see fnn func)  
	graph (optional)--> plot the phase space (attractor) in 3D
	"""
	x=PhaseSpace(datas, m, Tao, graph) #构建一个m维度的Tao延时的信号
	ED2=dist(x.T)   #计算距离矩阵
	posD=np.triu_indices_from(ED2, k=1)#距离矩阵坐标
	ED=ED2[posD]    #拉长
	max_eps=np.max(ED)
	min_eps=np.min(ED[np.where(ED>0)])
	max_eps=np.exp(math.floor(np.log(max_eps)))
	n_div=int(math.floor(np.log(max_eps/min_eps)))
	n_eps=n_div+1
	eps_vec=range(n_eps)
	unos=np.ones([len(eps_vec)])*-1
	eps_vec1=max_eps*np.exp(unos*eps_vec-unos)
	Npairs=((len(x[1,:]))*((len(x[1,:])-1)))
	C_eps=np.zeros(n_eps)
 
	for i in eps_vec:
        	eps=eps_vec1[i]
        	N=np.where(((ED<eps) & (ED>0)))
        	S=len(N[0])
        	C_eps[i]=float(S)/Npairs #比例

	omit_pts=1 
	k1=omit_pts
	k2=n_eps-omit_pts
	xd=np.log(eps_vec1)
	yd=np.log(C_eps)
	xp=xd[k1:k2]
	yp=yd[k1:k2]
	p = np.polyfit(xp, yp, 1)
	return p[0]


def PhaseSpace(data, m, Tao, graph=False):
  """
  Compute the phase space (attractor) a time series data with a time-lag Tao and an embedding dimension m
  data--> time series
  Tao--> time lag computed using the first zero crossing of the auto-correlation function (see Tao func)   
  m--> embeding dimension of the time-series, computed using the false neighbors method (see fnn func)  
  graph (optional)--> plot the phase space (attractor)
  """		
  ld=len(data)
  x = np.zeros([m, (ld-(m-1)*Tao)])
  for j in range(m):
      l1=(Tao*(j))
      l2=(Tao*(j)+len(x[1,:]))
      x[j,:]=data[l1:l2]
  if graph:
     fig = plt.figure()
     if m>2:
         ax = fig.add_subplot(111, projection='3d')
         ax.plot(x[0,:], x[1,:], x[2,:])
     else:
         ax = fig.add_subplot(111)
         ax.plot(x[0,:], x[1,:])         
  return x


def Tao(data):
    """
    使用第一过零率标准计算时间序列数据的时滞以构建相空间
    信号可以往后推迟多久才会完全不相关
    Compute the time-lag of a time series data to build the phase space using the first zero crossing rate criterion
    data--> time series
    """    
    corr=np.correlate(data, data, mode="full") #https://blog.csdn.net/icameling/article/details/85238412线性相关的实际意义是，向量中的各个与向量等长的子向量与向量的相似程度
    corr=corr[len(corr)/2:len(corr)]
    tau=0
    j=0
    while (corr[j]>0):
      j=j+1
    tau=j
    return tau


def fnn(data, maxm):
    """
    嵌入维度是嵌入对象（例如混乱吸引子）所需的最小维度。换句话说，这是您从测量开始重建相位肖像的空间的最小尺寸，其中轨迹不会跨越自身，也就是验证确定性。
    Compute the embedding dimension of a time series data to build the phase space using the false neighbors criterion
    data--> time series
    maxm--> maximmum embeding dimension
    """    
    RT=15.0
    AT=2
    sigmay=np.std(data, ddof=1) #计算标准差
    nyr=len(data)
    m=maxm
    EM=lagmat(data, maxlag=m-1) #每列延时一个单位，一共有14个
    EEM=np.asarray([EM[j,:] for j in range(m-1, EM.shape[0])])
    embedm=maxm
    for k in range(AT,EEM.shape[1]+1):
        fnn1=[]
        fnn2=[]
        Ma=EEM[:,range(k)]#每行依次变多，当做特征，相当于往回看
        D=dist(Ma)  #逐个单元之间的欧拉距离
        for i in range(1,EEM.shape[0]-m-k):#某个元素与其他所有元素之间的欧拉距离
            #print D.shape            
            #print(D[i,range(i-1)])
            d=D[i,:]
            pdnz=np.where(d>0)
            dnz=d[pdnz]
            Rm=np.min(dnz)
            l=np.where(d==Rm)
            l=l[0]
            l=l[len(l)-1]
            if l+m+k-1<nyr:
                fnn1.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/Rm)
                fnn2.append(np.abs(data[i+m+k-1]-data[l+m+k-1])/sigmay)
        Ind1=np.where(np.asarray(fnn1)>RT)
        Ind2=np.where(np.asarray(fnn2)>AT)
        if len(Ind1[0])/float(len(fnn1))<0.1 and len(Ind2[0])/float(len(fnn2))<0.1:
            embedm=k
            break
    return embedm
"""
t=np.asarray(range(1000))/1000.0
x=np.sin(2*np.pi*10*t)+np.sin(2*np.pi*100*t)+t*np.sin(2*np.pi*30*t)
m=fnn(x, 15)
tau=Tao(x)
cd=Dim_Corr(x, tau, m, True)
print ('embeding dimension='+str(m))
print ('time-lag='+str(tau))
print ('correlation dimension='+str(cd))

"""