# -- coding: utf-8 --
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#将a，b写成3x1的向量，三维的向量
"""
a = np.array([[1],[2],[3]])
b = np.array([[1],[1],[1]])

aa=np.linalg.inv(a.T.dot(a))
x = aa.dot(a.T).dot(b)
al=a.dot(x)
oc=b-al

xx=np.linspace(-0.5,1,10)
xx.shape=(1,10)
xxx=a.dot(xx)

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot(xxx[0,:],xxx[1,:],xxx[2,:],label='line a')
ax.plot(a[0],a[1],a[2],'r-o',label='a')
ax.plot([0,b[0]],[0,b[1]],[0,b[2]],'m-o',label="0B")
ax.plot([b[0][0],al[0][0]],[b[1][0],al[1][0]],[b[2][0],al[2][0]],'g-o',label="bP")
ax.plot([0,oc[0]],[0,oc[1]],[0,oc[2]],'y-o',label=u'Find Base coordinates')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.legend(loc='upper left')
ax.axis('equal')
plt.show()
"""
ox = np.array([[1],[2],[3]])
oy = np.array([[1],[1],[1]])
axis = np.array([ox[1]*oy[2]-ox[2]*oy[1],
                    ox[2]*oy[0]-ox[0]*oy[2],
                    ox[0]*oy[1]-ox[1]*oy[0]])
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot([0,ox[0]],[0,ox[1]],[0,ox[2]],'m-o',label="0x")
ax.plot([0,oy[0]],[0,oy[1]],[0,oy[2]],'y-o',label="0y")
ax.plot([0,axis[0]],[0,axis[1]],[0,axis[2]],'b-o',label="0a")
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
ax.legend(loc='upper left')
ax.axis('equal')
plt.show()