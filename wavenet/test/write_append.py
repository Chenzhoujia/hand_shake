# -*- coding: utf-8 -*-
import numpy as np
file = r'/home/chen/Documents/tensorflow-wavenet-master/analysis/static/test.txt'
a = np.array([1.1,2.1,3.1])
for i in range(10):
    if i == 0:
        np.savetxt(file,a*i)
    else:
        a_ = np.loadtxt(file)
        a_ = np.vstack((a_, a*i))
        np.savetxt(file, a_)