# -- coding:utf-8 --
import os,fnmatch
import numpy as np
import matplotlib.pyplot as plt

def load_generated_tra(directory):
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.txt'):
            files.append(os.path.join(root, filename))
    return files
data_path  = "/home/chen/Documents/tensorflow-wavenet-master/analysis/test/3"
files = load_generated_tra(data_path)
files.sort()
analysis = np.zeros((100,2))
for file_name_i, file_name in enumerate(files):
    file_data = np.loadtxt(file_name)
    file_data_mean = file_data.mean(axis = 0)
    analysis[file_name_i,:] = file_data_mean

fig = plt.figure(1)
fig.clear()
ax1 = plt.subplot(111)
analysis_x = np.linspace(0,np.size(analysis, axis=0)-1,np.size(analysis, axis=0))
ax1.plot(analysis_x,analysis[:,0] , linewidth=1)
#ax1.plot(analysis_x,analysis[:,1] , linewidth=0.3)
ax1.set_title(analysis[:,0].mean())
ax1.grid(True)
plt.savefig(data_path + "/static" + ".jpg")
