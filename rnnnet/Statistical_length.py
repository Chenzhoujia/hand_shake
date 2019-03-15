import os,fnmatch
import numpy as np

def load_generated_tra(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = []
    files_out = []
    min = 40000
    i = 0
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.txt'):
            i = i+1

            files.append(os.path.join(root, filename))
            filename =os.path.join(root, filename)
            onefile_randomized_files = np.loadtxt(filename)
            length_now = np.size(onefile_randomized_files, 0)
            if length_now < min:
                min = length_now
                print(np.size(onefile_randomized_files,0))
                print(filename)
                print(i)
            if length_now<3000:
                files_out.append(filename)

    return files_out

files_out = load_generated_tra("/media/chen/4CBEA7F1BEA7D1AE/Download/hand_dataset/pakinson/shake_tra_test", 0)
file = open('file_out_test_name.txt','w')
file.write(str(files_out))
file.close()
print(files_out)
"""

import sys
result=[]
with open('file_out_name.txt','r') as f:
    for line in f:
        line = line[1:-1]
        lines = line.split(',')

        for file in lines:
            file = file.strip()
            file = file[1:-1]
            print(file)
            onefile_randomized_files = np.loadtxt(file)
            print(np.shape(onefile_randomized_files))
"""
