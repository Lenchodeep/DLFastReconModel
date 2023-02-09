import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import pandas as pd
import numpy as np
from os import path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.ticker as ticker
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
        row[2] = float(row[2].strip())
        y.append((row[2]))
        x.append((row[1]))

    return x, y


def DDBAblationPSNR():
    folderpath = 'E:/DLCode/DLFastReconstruction/SepGAN/curves/DRDBEXPERIMENT/'
    filename0 = 'PSNR0.csv'
    filename1 = 'PSNR1.csv'
    filename2 = 'PSNR2.csv'
    filename4 = 'PSNR3.csv'
    filename5 = 'PSNR4.csv'

    # smooth(folderpath, filename, 0.5)


    x0, y0 = readcsv(path.join(folderpath,filename0))
    x1, y1 = readcsv(path.join(folderpath,filename1))
    x2, y2 = readcsv(path.join(folderpath,filename2))
    x4, y4 = readcsv(path.join(folderpath,filename4))
    x5, y5 = readcsv(path.join(folderpath,filename5))

     

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x0,y0,marker = 'o', markersize = 3,color='#BF6F6D', label = 'n=0')        
    ax.plot(x1,y1,marker = '^', markersize = 3,color='#589B90', label = 'n=1')
    ax.plot(x2,y2,marker = '*', markersize = 3,color='#3570A2', label = 'n=2')
    ax.plot(x4,y4,marker = 'v', markersize = 3,color='#C49514', label = 'n=3')
    ax.plot(x5,y5,marker = 's', markersize = 3,color='#7D4892', label = 'n=4')
    ax.add_patch(
     patches.Rectangle(
        (60, 40.2),
        20,
        1,
        edgecolor = 'red',
        facecolor = 'white',
        fill=True
     ) )
    plt.tick_params(labelsize=16)
    plt.grid(ls='--')
    plt.xlabel('epoch', fontdict = {'size':18})
    plt.ylabel('Average PSNR', fontdict = {'size':18})
    plt.ylim((36, 41.5))
    plt.legend(loc='lower right', prop = {'size':16})
    plt.savefig(path.join(folderpath,'PSNR.png'),dpi=600)
    plt.show()

    x0_zoom = x0[59:]
    for i in range(len(x0_zoom)):
        x0_zoom[i] = x0_zoom[i]+1
    y0_zoom = y0[59:]
    x1_zoom = x1[59:]
    for i in range(len(x1_zoom)):
        x1_zoom[i] = x1_zoom[i]+1

    y1_zoom = y1[59:]
    x2_zoom = x2[59:]
    for i in range(len(x2_zoom)):
        x2_zoom[i] = x2_zoom[i]+1

    y2_zoom = y2[59:]
    x4_zoom = x4[59:]
    for i in range(len(x4_zoom)):
        x4_zoom[i] = x4_zoom[i]+1

    y4_zoom = y4[59:]
    x5_zoom = x5[59:]
    for i in range(len(x5_zoom)):
        x5_zoom[i] = x5_zoom[i]+1

    y5_zoom = y5[59:]

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.plot(x0_zoom,y0_zoom,marker = 'o', markersize = 3, color='#BF6F6D', label = 'n=0')     
    ax1.plot(x1_zoom,y1_zoom,marker = '^', markersize = 3, color='#589B90', label = 'n=1')
    ax1.plot(x2_zoom,y2_zoom,marker = '*', markersize = 3, color='#3570A2', label = 'n=2')
    ax1.plot(x4_zoom,y4_zoom,marker = 'v', markersize = 3, color='#C49514', label = 'n=3')
    ax1.plot(x5_zoom,y5_zoom,marker = 's', markersize = 3, color='#7D4892', label = 'n=4')
    plt.grid(ls='--')
    plt.tick_params(labelsize=16)
    plt.xlabel('epoch', fontdict = {'size':18})
    plt.ylabel('Average PSNR', fontdict = {'size':18})
    for axis in [ax1.xaxis, ax1.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylim((39.5, 41.5))
    plt.xlim((60,81))
    plt.legend(loc='lower right', prop = {'size':16})
    plt.savefig(path.join(folderpath,'PSNR_ZOOM.png'),dpi=600)
    plt.show()


def DDBAblationNMSE():
    folderpath = 'E:/DLCode/DLFastReconstruction/SepGAN/curves/DRDBEXPERIMENT/'
    filename0 = 'NMSE0.csv'
    filename1 = 'NMSE1.csv'
    filename2 = 'NMSE2.csv'
    filename4 = 'NMSE3.csv'
    filename5 = 'NMSE4.csv'

    # smooth(folderpath, filename, 0.5)


    x0, y0 = readcsv(path.join(folderpath,filename0))
    x1, y1 = readcsv(path.join(folderpath,filename1))
    x2, y2 = readcsv(path.join(folderpath,filename2))
    x4, y4 = readcsv(path.join(folderpath,filename4))
    x5, y5 = readcsv(path.join(folderpath,filename5))

     

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x0,y0,marker = 'o', markersize = 3,color='#BF6F6D', label = 'n=0')        
    ax.plot(x1,y1,marker = '^', markersize = 3,color='#589B90', label = 'n=1')
    ax.plot(x2,y2,marker = '*', markersize = 3,color='#3570A2', label = 'n=2')
    ax.plot(x4,y4,marker = 'v', markersize = 3,color='#C49514', label = 'n=3')
    ax.plot(x5,y5,marker = 's', markersize = 3,color='#7D4892', label = 'n=4') 
    ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')

    ax.add_patch(
     patches.Rectangle(
        (60, 0.0013),
        20,
        0.0005,
        edgecolor = 'red',
        facecolor = 'white',
        fill=True
     ) )
    plt.tick_params(labelsize=16)
    plt.grid(ls='--')
    plt.xlabel('epoch', fontdict = {'size':18})
    plt.ylabel('Average NMSE (×$10^{-3}$)', fontdict = {'size':18})
    plt.ylim((0.0010, 0.0040))
    plt.legend(loc='upper right', prop = {'size':16})
    plt.savefig(path.join(folderpath,'NMSE.png'),dpi=600)
    plt.show()

    x0_zoom = x0[59:]
    for i in range(len(x0_zoom)):
        x0_zoom[i] = x0_zoom[i]+1
    y0_zoom = y0[59:]
    x1_zoom = x1[59:]
    for i in range(len(x1_zoom)):
        x1_zoom[i] = x1_zoom[i]+1

    y1_zoom = y1[59:]
    x2_zoom = x2[59:]
    for i in range(len(x2_zoom)):
        x2_zoom[i] = x2_zoom[i]+1

    y2_zoom = y2[59:]
    x4_zoom = x4[59:]
    for i in range(len(x4_zoom)):
        x4_zoom[i] = x4_zoom[i]+1

    y4_zoom = y4[59:]
    x5_zoom = x5[59:]
    for i in range(len(x5_zoom)):
        x5_zoom[i] = x5_zoom[i]+1

    y5_zoom = y5[59:]

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    ax1.plot(x0_zoom,y0_zoom,marker = 'o', markersize = 3, color='#BF6F6D', label = 'n=0')     
    ax1.plot(x1_zoom,y1_zoom,marker = '^', markersize = 3, color='#589B90', label = 'n=1')
    ax1.plot(x2_zoom,y2_zoom,marker = '*', markersize = 3, color='#3570A2', label = 'n=2')
    ax1.plot(x4_zoom,y4_zoom,marker = 'v', markersize = 3, color='#C49514', label = 'n=3')
    ax1.plot(x5_zoom,y5_zoom,marker = 's', markersize = 3, color='#7D4892', label = 'n=4')
    plt.tick_params(labelsize=16)
    plt.grid(ls='--')
    plt.xlabel('epoch', fontdict = {'size':18})
    plt.ylabel('Average NMSE (×$10^{-3}$)', fontdict = {'size':18})
    for axis in [ax1.xaxis, ax1.yaxis]:
        axis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.ylim((0.00110,0.00180))
    plt.xlim((60,81))
    plt.legend(loc='lower right', prop = {'size':16})
    plt.savefig(path.join(folderpath,'NMSE_ZOOM.png'),dpi=600)
    plt.show()


def DCpsnr():
    folderpath = 'E:/DLCode/DLFastReconstruction/PsychoGanCode/curves/DCexperiment/'
    filename0 = 'oursPsnr.csv'
    filename1 = 'noDCPSNR.csv'
    filename2 = 'noDCPSNR-Val.csv'
    x0, y0 = readcsv(path.join(folderpath,filename0))
    x1, y1 = readcsv(path.join(folderpath,filename1))
    x2, y2 = readcsv(path.join(folderpath,filename2))

    x0 = [i+1 for i in x0]
    x1 = [i+1 for i in x1]
    x2 = [i+1 for i in x2]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x0,y0,marker = 'o', markersize = 3,color='#BF6F6D', label = 'SepGAN')        
    ax.plot(x1,y1,marker = '^', markersize = 3,color='#589B90', label = 'SepGAN-G')
    # ax.plot(x2,y2,marker = '*', markersize = 3,color='#3570A2', label = 'val')

    plt.xlabel('epoch', fontdict = {'size':14})
    plt.ylabel('Average PSNR', fontdict = {'size':14})
    plt.ylim((30, 41.5))
    plt.legend(loc='lower right', prop = {'size':14})
    plt.savefig(path.join(folderpath,'PSNRCompare.png'))
    plt.show()

def DCnmse():
    folderpath = 'E:/DLCode/DLFastReconstruction/PsychoGanCode/curves/DCexperiment/'
    filename0 = 'oursNMSE.csv'
    filename1 = 'noDCNMSE.csv'
    x0, y0 = readcsv(path.join(folderpath,filename0))
    x1, y1 = readcsv(path.join(folderpath,filename1))
    
    x0 = [i+1 for i in x0]
    x1 = [i+1 for i in x1]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x0,y0,marker = 'o', markersize = 3,color='#BF6F6D', label = 'SepGAN')        
    ax.plot(x1,y1,marker = '^', markersize = 3,color='#589B90', label = 'SepGAN-G')
        # ax.plot(x2,y2,marker = '*', markersize = 3,color='#3570A2', label = 'val')

    plt.xlabel('epoch', fontdict = {'size':14})
    plt.ylabel('Average NMSE', fontdict = {'size':14})
    plt.ylim((0,0.01 ))
    plt.legend(loc='upper right', prop = {'size':14})
    plt.savefig(path.join(folderpath,'NMSECompare.png'))
    plt.show()

def klossNMSE():
    folderpath = 'E:/DLCode/DLFastReconstruction/SepGAN/curves/kloss/'
    filename0 = 'nmse2.csv'
    filename1 = 'NMSE.csv'
    x0, y0 = readcsv(path.join(folderpath,filename0))
    x1, y1 = readcsv(path.join(folderpath,filename1))
    
    x0 = [i+1 for i in x0]
    x1 = [i+1 for i in x1]
    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x0,y0,marker = 'o', markersize = 3,color='#BF6F6D', label = 'FFL')        
    ax.plot(x1,y1,marker = '*', markersize = 3,color='#3570A2', label = 'MSE')
    ax.ticklabel_format(style='sci', scilimits=(0,2), axis='y')
    plt.tick_params(labelsize=16)
    plt.grid(ls='--')
    plt.xlabel('epoch', fontdict = {'size':18})
    plt.ylabel('Average NMSE (×$10^{-3}$)', fontdict = {'size':18})
    plt.ylim((0.001,0.004 ))
    plt.legend(loc='upper right', prop = {'size':16})
    plt.savefig(path.join(folderpath,'NMSECompare.png'),dpi=600)
    plt.show()

def klossPSNR():
    folderpath = 'E:/DLCode/DLFastReconstruction/SepGAN/curves/kloss/'
    filename0 = 'psnr2.csv'
    filename1 = 'PSNR.csv'
    x0, y0 = readcsv(path.join(folderpath,filename0))
    x1, y1 = readcsv(path.join(folderpath,filename1))

    x0 = [i+1 for i in x0]
    x1 = [i+1 for i in x1]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.plot(x0,y0,marker = 'o', markersize = 3,color='#BF6F6D', label = 'FFL')        
    ax.plot(x1,y1,marker = '*', markersize = 3,color='#3570A2', label = 'NMSE')
    plt.tick_params(labelsize=16)

    plt.grid(ls='--')
    plt.xlabel('epoch', fontdict = {'size':18})
    plt.ylabel('Average PSNR', fontdict = {'size':18})
    plt.ylim((30, 41.5))
    plt.legend(loc='lower right', prop = {'size':16})
    plt.savefig(path.join(folderpath,'PSNRCompare.png'),dpi=600)
    plt.show()

klossNMSE()
