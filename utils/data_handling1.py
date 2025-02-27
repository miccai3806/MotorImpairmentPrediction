"""
Created on 2025
@author: Dr. XXXXX
"""
import keras
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import glob
import random
import numpy as np
from skimage import io
import cv2
import pandas as pd 
from natsort import natsorted

# The libraries of reading nifti files
import os
import nibabel as nib
from nibabel.testing import data_path
from numpy.random import seed # to generate random indexes of patient files
import sys 
sys.path.append(r".\utils")
from Data_Generator1 import *

###################### ENSEMBLE MODELS ########################################
def ensemble_datagenerator(training_listAll, training_demlist, ytraining_list,
                           validation_listAll, validation_demlist, yvalidation_list,
                           prediction_listAll, prediction_demlist, yprediction_list,
                           batch_size,input_shape,input_dem,n_channels,num_classes,shuffle):
         tr_generator = train_generator(training_listAll, training_demlist, ytraining_list, batch_size,input_shape,input_dem,
                                           n_channels,num_classes,shuffle) 
              
         step_size_train =  (np.ceil(len(training_listAll[0])/ float(batch_size))).astype(np.int32)
        
         val_generator =validation_generator(validation_listAll, validation_demlist, yvalidation_list, batch_size,input_shape,
                                                 input_dem, n_channels,num_classes,shuffle)
         step_size_val =(np.ceil(len(validation_listAll[0])/float(batch_size))).astype(np.int32)  
         
         
         test_generator =prediction_generator(prediction_listAll, prediction_demlist, yprediction_list, batch_size,input_shape,
                                                  input_dem, n_channels,num_classes,shuffle)
         step_size_test=(np.ceil(len(prediction_listAll[0])/float(batch_size))).astype(np.int32)  
         
         return  tr_generator,  step_size_train,  val_generator, step_size_val, test_generator, step_size_test


def ensemble_data(datasetpath,cvpath,num_classes,strokeclasses,valbool,trainrate,testrate,valrate,imageclasses):
        datasetpath=datasetpath+'\\'+cvpath    
        training_demlist,prediction_demlist,validation_demlist=[],[],[]
        ytraining_list, yprediction_list, yvalidation_list=[],[],[] 
        training_listAll,prediction_listAll,validation_listAll=[],[],[]
        
        #--------Read the outputs-the motor function classes---------------------------
        if num_classes==2:
            col_name='J'
            col=9
        elif num_classes==3:
            col_name='K'
            col=10
        elif num_classes==4:
            col_name='L'
            col=11
        elif num_classes==5:
            col_name='M'
            col=12
            
        patient_output = pd.read_excel(datasetpath+'\Motor_Scores_All3.xlsx',sheet_name=0,
                        header=None, usecols=[0, 2, 3, 4, col], names=['A','C','D','E', col_name],skiprows=1)
        patient_output["C"]=patient_output["C"]/patient_output["C"].max() # age normalization
        patient_output["E"]=patient_output["E"]/patient_output["E"].max() #stroke time normalization
        
        patient_list= patient_output.values.tolist() 
        #-----------------------------------------------------------------------------
        
        # Prepare Lists---------------------------------------------------------------
        training_listAll,prediction_listAll,validation_listAll,ytraining_list,yprediction_list,yvalidation_list,training_demlist,prediction_demlist,validation_demlist=get_list2(strokeclasses,shuffle,datasetpath,valbool,trainrate,testrate,valrate,imageclasses,patient_list)
        ct=1
        training_listAll,ytraining_list,training_demlist=get_chlist2(training_listAll,len(imageclasses),ytraining_list,training_demlist,ct) 
        
        if testrate>0.0:
            ct=0
            prediction_listAll,yprediction_list,prediction_demlist=get_chlist2(prediction_listAll,len(imageclasses),yprediction_list,prediction_demlist,ct)
        else:
            prediction_listAll=[]
        if valbool==1: 
            ct=0
            validation_listAll,yvalidation_list,validation_demlist=get_chlist2(validation_listAll,len(imageclasses),yvalidation_list,validation_demlist,ct)     
        else:
            validation_listAll=[]
        #-----------------------------------------------------------------------------    
        print("Training dataset sample=",len(ytraining_list))  
        print("Testing dataset sample=",len(yprediction_list)) 
        print("Validation dataset sample",len(yvalidation_list)) 
        return  training_listAll, training_demlist, ytraining_list,validation_listAll,validation_demlist, yvalidation_list,prediction_listAll,prediction_demlist, yprediction_list
###############################################################################

# output 
def get_output2(tclass,item,patient_list,s):  
    temp = os.path.splitext(item)[0]  
    temp = os.path.basename(temp)
    temp, ext = os.path.splitext(temp)
    output=0.0    
    uz=int(np.size(patient_list)/s)
    for av in range (0,uz):
        ss=patient_list[av][0]+'_'
        if temp.find(ss)!=-1:
            output=round(patient_list[av][s-1], 2) 
            return output 

# demographic data
def get_dem_output2(tclass,item,patient_list,s):  
    temp = os.path.splitext(item)[0]  
    temp = os.path.basename(temp)
    temp, ext = os.path.splitext(temp)

    total_n=int(np.size(patient_list)/s)
    uz=int(np.size(patient_list)/s)
    for av in range (0,uz):
        ss=patient_list[av][0]+'_'
        if temp.find(ss)!=-1:
            output=[]
            output.append(patient_list[av][1]) # age
            output.append(patient_list[av][2]) # gender
            output.append(patient_list[av][3]) # stroke_time 
            return output

def get_chlist2(listAll,n,ytraining_list,training_demlist,ct):  
    dim=len(listAll)
    swap=[]
    listAll2=[]
    for i in range (0,n):
        if listAll[i]:
            temp = listAll[i][0]
            temp = os.path.basename(temp)
            temp, ext = os.path.splitext(temp)
            temp=temp.split('_')
            uz1=np.size(temp)
            if temp[0]=='n1': # for DTI maps folder
                if uz1==5:
                    temp=temp[3]
                elif uz1==6:
                    if temp[5]=='reg.nii':
                        temp=temp[4]
                    else:
                        temp=temp[3]        
                elif uz1==7:
                    temp=temp[4]   
            if temp[0]=='t1': # for MRI folder                         
                if uz1==4:
                    if temp[3]=='WM' or temp[3]=='WM.nii':   
                        temp='WM'
                    elif temp[3]=='GM' or temp[3]=='GM.nii': 
                        temp='GM'
                    elif temp[3]=='MRI.nii' or temp[3]=='MRI.nii': 
                        temp='MRI'
                    else:
                        temp=temp[3]
                elif uz1==5:
                    if temp[4]=='WM' or temp[4]=='WM.nii':   
                        temp='WM'
                    elif temp[4]=='GM' or temp[4]=='GM.nii': 
                        temp='GM'
                    elif temp[4]=='MRI.nii' or temp[4]=='MRI.nii': 
                        temp='MRI'
                    else:
                        temp=temp[3]                
                elif uz1==6:                       
                    temp=temp[4] 
                    
            counter=0
                                
            swap=[]
            swap.extend(listAll[i])        
            for j in range (i+1,dim):
                if listAll[j]:
                    temp1 = listAll[j][0]
                    temp1 = os.path.basename(temp1)
                    temp1, ext = os.path.splitext(temp1)
                    temp1=temp1.split('_')
                    uz2=np.size(temp1)
                    
                    if temp1[0]=='n1': # for DTI maps folder
                        if uz2==5:
                            temp1=temp1[3]
                        elif uz2==6:
                            if temp1[5]=='reg.nii':
                                temp1=temp1[4]
                            else:
                                temp1=temp1[3]        
                        elif uz2==7:
                            temp1=temp1[4] 
                            
                    elif temp1[0]=='t1': # for MRI images folder                         
                        if uz2==4:
                            if temp1[3]=='WM' or temp1[3]=='WM.nii':   
                                temp1='WM'
                            elif temp1[3]=='GM' or temp1[3]=='GM.nii': 
                                temp1='GM'
                            elif temp1[3]=='MRI' or temp1[3]=='MRI.nii': 
                                temp1='MRI'    
                            else:
                                 temp1=temp1[3]   
                        elif uz2==5:
                            if temp1[4]=='WM' or temp1[4]=='WM.nii':   
                                temp1='WM'
                            elif temp1[4]=='GM' or temp1[4]=='GM.nii': 
                                temp1='GM'
                            elif temp1[4]=='MRI' or temp1[4]=='MRI.nii': 
                                temp1='MRI'    
                            else:
                                 temp1=temp1[3]   
                        elif uz2==6:                        
                            temp1=temp1[4]  # find folder name
                    
                    if temp==temp1:
                        swap.extend(listAll[j])
                        listAll[j]=[]
            listAll2.append(swap)
            
            
    #Shuffle List  
    if ct==1:
        list2_IDs=listAll2[0] # control the batch size
        indexes = np.arange(len(list2_IDs))    
        np.random.shuffle(indexes) # shuffle the files indexes
    elif ct==0:
        list2_IDs=listAll2[0] # control the batch size
        indexes = np.arange(len(list2_IDs))
        
    dim=len(listAll2)
    ytraining_list=[ytraining_list[k] for k in indexes]
    training_demlist=[training_demlist[k] for k in indexes]
    
    for i in range(0,dim): 
        list2_IDs = [listAll2[i][k] for k in indexes]         
        listAll2[i]= list2_IDs  
                   
    return listAll2, ytraining_list,training_demlist
             
def get_files3(training_index, prediction_index, validation_index,filepath,strokeclass,imageclass): 
    files = natsorted(glob.glob(filepath+"\\Train\\{0}\\{1}\\*".format(strokeclass, imageclass)))    
    files2 = natsorted(glob.glob(filepath+"\\Test\\{0}\\{1}\\*".format(strokeclass, imageclass)))
    
    #training,prediction,validation=[],[],[]
    training = [files[k] for k in training_index]        
    prediction = [files2[k] for k in prediction_index]      
    validation = [files2[k] for k in validation_index]      
    return training, prediction, validation # return file names   

def get_indexes2(shuffle,filepath, valbool,trainrate,testrate,valrate,strokeclass,imageclass):    
    files = glob.glob(filepath+"\\Train\\{0}\\{1}\\*".format(strokeclass, imageclass))
    indexes = np.arange(len(files))  
    
    files2 = glob.glob(filepath+"\\Test\\{0}\\{1}\\*".format(strokeclass, imageclass))
    indexes2 = np.arange(len(files2))  
    
    training, validation, prediction=[],[],[]    
    if shuffle == True:
            indexes=np.random.shuffle(indexes)
    if valbool==1:
        training = indexes 
        validation = indexes2
        prediction = indexes2
    elif valbool==0:
        training = indexes
        prediction= indexes2     
        validation=[]
    return training, prediction, validation

def get_list2(strokeclasses,shuffle,datasetpath,valbool,trainrate,testrate,valrate,imageclasses,patient_list):    
    training_demlist,prediction_demlist,validation_demlist=[],[],[]
    ytraining_list, yprediction_list, yvalidation_list=[],[],[] 
    training_listAll,prediction_listAll,validation_listAll=[],[],[]  
    
    # Prepare the lists----------------------------------------------------------- 
    for sk in range(0,len(strokeclasses)): 
        tclass=strokeclasses[sk]
        # get indexes of first folder
        training_index, prediction_index, validation_index=get_indexes2(shuffle,datasetpath, valbool,trainrate,testrate,valrate,tclass,imageclasses[0])
        control=0
        for rk in imageclasses:
            training_list, prediction_list, validation_list=[], [], []
            training, prediction, validation = get_files3(training_index, prediction_index,validation_index, datasetpath,tclass,rk)
            for item in training:
                training_list.append(item) 
                if control==0:
                    training_demlist.append(get_dem_output2(tclass,item,patient_list,5))           
                    ytraining_list.append(get_output2(tclass,item,patient_list,5)) 
            for item in prediction:
                prediction_list.append(item)
                if control==0:
                    prediction_demlist.append(get_dem_output2(tclass,item,patient_list,5))                     
                    yprediction_list.append(get_output2(tclass,item,patient_list,5)) 
            for item in validation:
                validation_list.append(item) 
                if control==0:
                    validation_demlist.append(get_dem_output2(tclass,item,patient_list,5))        
                    yvalidation_list.append(get_output2(tclass,item,patient_list,5)) 
            training_listAll.append(training_list) 
            prediction_listAll.append(prediction_list) 
            validation_listAll.append(validation_list)
            control=control+1 
    return training_listAll,prediction_listAll,validation_listAll,ytraining_list,yprediction_list,yvalidation_list,training_demlist,prediction_demlist,validation_demlist


