#!/usr/bin/python
#Checklist for operation:
#1. Change "Suffix" to appropriate ending from runs.
#2. Change the booleans on the FLAG list below to correspond with which BAG files are available. Adjust as necessary.
#3. Comment or otherwise disable any tests you do not wish to run in their corresponding cells.

#This file will require updating if rostopics are moved from one bag file to another as the references are hardcoded.
import matplotlib # if not installed sudo apt-get install python3-matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import subprocess
import tf
import math
import csv
import sys
import ast
import os
import numpy as np
from datetime import datetime
from struct import *
from mpl_toolkits.mplot3d import Axes3D
from LatLongUTMconversion import LLtoUTM
csv.field_size_limit(sys.maxsize)

pylab.rcParams['figure.figsize'] = 15, 8  # that's default image size for this interactive session

#Every BAG file should have this suffix. Example: other2016-05-26-10-34-51.bag
suffix = "2017-12-22-02-55-41"
suffix = "2017-12-22-02-52-14"

#The folder the BAG files are in:
bagfolder = "Data/"

#Prefix of bag files if used
bagfile = suffix + ".bag"

print("===== Setup Finished")

## ======== Convert topics in a bagfile into a CSV file =================
#CREATE_CSV = True  # Only need to run this once per bagfile
CREATE_CSV = False
#CONVERSION FROM .BAG to .CSV, attempts to check each bagfile accordingly.
print(bagfolder,bagfile)
if CREATE_CSV == True:
    try:
        returnCode = subprocess.call(['./bag2csv_v2.py', bagfolder, bagfile])
        print("Return Code: %d" %returnCode)
    except:
        print("Unexpected error parsing bagfile into csv:", sys.exc_info()[0])
        pass

print("===== Done parsing bagfiles into CSV files")
## ======================================================================

## =============== Parsing IMU Orientation  =================
IMU_PARSE = True
if IMU_PARSE:

    # Get cns5000 data from CSV files
    cns5000DataList = []
    with open(bagfolder + bagfile[:-4] + "/_slash_cns5000_slash_imu_slash_raw.csv",'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            cns5000DataList.append(row)
    cns5000DataListColumnnames = cns5000DataList.pop(0)
    Orientdataex = cns5000DataList[10]
    print("CNS:",cns5000DataListColumnnames)
    #print(Orientdataex)
    cns5000Time=[]
    cns5000roll=[]
    cns5000pitch=[]
    cns5000yaw=[]
    cns5000x=[]
    cns5000y=[]
    cns5000z=[]
    x_pos=0
    y_pos=0
    lastTime=0
    for row in cns5000DataList:
        cns5000Time.append(int(row[0])*0.0000000001)
        r,p,yaw = tf.transformations.euler_from_quaternion([float(row[8]),float(row[9]),float(row[10]),float(row[11])])
        cns5000roll.append(r)
        cns5000pitch.append(p)
        cns5000yaw.append(yaw)
        x,y,z = float(row[9]),float(row[10]),float(row[11])
        dt=(int(row[0])-lastTime)*0.001
        print("dt",dt)
        dx = x * dt * math.cos(yaw)
        dy = y * dt * math.sin(yaw)
        x_pos += dx
        y_pos += dy         
        cns5000x.append(x_pos)
        cns5000y.append(y_pos)
        lastTime=int(row[0])
        
    # Get xsens 300 data from CSV files
    x300DataList = []
    with open(bagfolder + bagfile[:-4] + "/_slash_xsens300_slash_imu_slash_data.csv",'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x300DataList.append(row)
    x300DataListColumnnames = x300DataList.pop(0)
    Orientdataex = x300DataList[10]
    print("X300:",x300DataListColumnnames)
    #print(Orientdataex)
    x300Time=[]
    x300roll=[]
    x300pitch=[]
    x300yaw=[]
    for row in x300DataList:
        x300Time.append(int(row[0])*0.0000000001)
        r,p,y = tf.transformations.euler_from_quaternion([float(row[8]),float(row[9]),float(row[10]),float(row[11])])
        x300roll.append(r)
        x300pitch.append(p)
        #y=y*180/3.141
        #if y < 0:
        #    y = y + 6.282
        #y = y -2.356
        y=y-3.141+0.75
        if y<-3.141:
            y=y+6.282
        x300yaw.append(y)

    # Get xsens 700 data from CSV files
    x700DataList = []
    with open(bagfolder + bagfile[:-4] + "/_slash_xsens700_slash_imu_slash_data.csv",'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            x700DataList.append(row)
    x700DataListColumnnames = x700DataList.pop(0)
    Orientdataex = x700DataList[10]
    print("x700:",x700DataListColumnnames)
    #print(Orientdataex)
    x700Time=[]
    x700roll=[]
    x700pitch=[]
    x700yaw=[]
    for row in x700DataList:
        x700Time.append(int(row[0])*0.0000000001)
        r,p,y = tf.transformations.euler_from_quaternion([float(row[8]),float(row[9]),float(row[10]),float(row[11])])
        x700roll.append(r)
        x700pitch.append(p)
        #y=y*180/3.141
        #y=y+1.57        
        y=y+0.75
        if y<-3.141:
            y=y+6.282
        x700yaw.append(y)

    print("len(x700yaw)")
    print(len(x700Time))
    
    # Get MoCap data from CSV files
    mcapDataList = []
    with open(bagfolder + bagfile[:-4] + "/_slash_vrpn_client_node_slash_imu_stack_slash_pose.csv",'rb') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            mcapDataList.append(row)
            
    mcapDataListColumnnames = mcapDataList.pop(0)
    Orientdataex = mcapDataList[10]
    print("mocap:",mcapDataListColumnnames)
    #print(Orientdataex)
    mcapTime=[]
    mcaproll=[]
    mcappitch=[]
    mcapyaw=[]
    mcapx=[]
    mcapy=[]
    mcapz=[]
    for row in mcapDataList:
        mcapTime.append(int(row[0])*0.0000000001)
        r,p,y = tf.transformations.euler_from_quaternion([float(row[13]),float(row[14]),float(row[15]),float(row[16])])
        mcaproll.append(r)
        mcappitch.append(p)
        #y=y*180/3.141
        #if y < -1.5:
        #    y=y+6.282
        mcapyaw.append(y)
        x,y,z = float(row[9]),float(row[10]),float(row[11])
        mcapx.append(x)
        mcapy.append(y)
        mcapz.append(z)

    import matplotlib.patches as mpatches

    #print("len(mcapyaw)")
    #print(len(mcapTime))
    SHOWYAW=False
    if SHOWYAW:
        plt.figure(1)
        plt.title('Yaw of Imu Stack')
        plt.xlabel('Ros Time (seconds)')
        plt.ylabel('Yaw (radians)')
        p1,=plt.plot(x700Time,x700yaw, 'g.')
        p2,=plt.plot(x300Time,x300yaw, 'b.')
        p3,=plt.plot(cns5000Time,cns5000yaw, 'r.')
        p4,=plt.plot(mcapTime,mcapyaw, 'c.')
        plt.legend((p1,p2,p3,p4),("xsens700","xsens300","CNS5000","Mocap"), 'upper left')  
    plt.figure(2)
    p5,=plt.plot(mcapx,mcapy, 'c.')
    plt.figure(3)
    p6,=plt.plot(cns5000Time,cns5000x, 'r.')
    plt.figure(4)
    p7,=plt.plot(cns5000Time,cns5000y, 'b.')
    plt.figure(5)
    p8,=plt.plot(cns5000x,cns5000y, 'r.')
    
    
    
    plt.show()   
     
   

