import os
import sys


myfile = "./data/test_annotation.txt" 
outfile = "./data/test_annotation_pai.txt"

with open(myfile, 'r+') as f:
    temp = f.read().splitlines()

new_temp = [path.replace('datasets', 'vocdatasets') + '\n' for path in temp]
# new_temp = [path.replace('./vocdatasets/VOCdevkit/VOC2007/JPEGImages/', '').split(".jpg")[0] + '\n' for path in temp]


with open(outfile, 'w') as f2:
    f2.writelines(new_temp)