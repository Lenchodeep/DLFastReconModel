import pandas as pd
import numpy as np
import os
from os import path
import matplotlib.pyplot as plt
import csv

def smooth(csv_path,csv_file, weight = 0.5):
    data = pd.read_csv(filepath_or_buffer=path.join(csv_path, csv_file), header=0,names=['Step', 'Value'], dtype={'Step':np.int16,'Value':np.float32})
    scalar = data['Value'].values
    last = scalar[0]
    smoothed = []
    for point in scalar:
        smoothed_val = last*weight + (1-weight)*point
        smoothed.append(smoothed_val)
        last = smoothed_val

    save = pd.DataFrame({'Step':data['Step'].values, 'Value':smoothed})
    save.to_csv(path.join(csv_path,('smoothed_'+csv_file)))

def readcsv(files):
    csvfile = open(files,'r')
    plots = csv.reader(csvfile)
    x=[]
    y=[]
    for row in plots:
        y.append((row[2]))
        x.append((row[1]))
    return x,y



def readcsv(files):
    csvfile = open(files, 'r')
    plots = csv.reader(csvfile, delimiter=',')
    x = []
    y = []
    step = 0
    for row in plots:
        # x.append(step)
        step = step+1
        row[1] = float(row[1].strip())
        print(row[2])
        row[2] = float(row[2].strip())
        y.append((row[2]))
        x.append((row[1]))

    return x, y


def main():
    folderpath = 'E:/DLCode/DLFastReconstruction/SigleModel/checkpoints/'
    filename0 = 'Loss0.csv'
    filename1 = 'Loss1.csv'
    filename2 = 'Loss2.csv'
    filename4 = 'Loss4.csv'

    # smooth(folderpath, filename, 0.5)


    x0, y0 = readcsv(path.join(folderpath,filename0))
    x1, y1 = readcsv(path.join(folderpath,filename1))
    x2, y2 = readcsv(path.join(folderpath,filename2))
    x4, y4 = readcsv(path.join(folderpath,filename4))


    plt.figure()

    plt.plot(x0,y0,color='green', label = 'p=0')        
    plt.plot(x1,y1,color='red', label = 'p=0.1')
    plt.plot(x2,y2,color='blue', label = 'p=0.2')
    plt.plot(x4,y4,color='orange', label = 'p=0.4')

    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.ylim((0,0.006))
    plt.legend(loc='upper left')
    plt.savefig(path.join(folderpath,'Loss.png'))
    plt.show()
main()
