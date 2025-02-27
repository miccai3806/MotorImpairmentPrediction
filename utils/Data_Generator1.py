# -*- coding: utf-8 -*-
"""
Created on 2025
@author: Dr. XXXX
"""
import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import nibabel as nib
from nibabel.testing import data_path
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence # thread
import threading

class ThreadSafeIterator:
    def __init__(self, it):
        self.it = it              
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def __next__(self):
        with self.lock:
            return self.it.__next__()


def threadsafe_generator(f):
    def g(*args, **kwargs):
        return ThreadSafeIterator(f(*args, **kwargs))
    return g


@threadsafe_generator
def train_generator(list_IDs, list_demIDs,labels, batch_size=10, dim=(182,218,182), dim2=(3,), n_channels=1,
                    n_classes=5, shuffle=False):
    while True:
        list2_IDs=list_IDs[0] # control the batch size
        indexes = np.arange(len(list2_IDs))
        
        if shuffle == True:
            np.random.shuffle(indexes)
            
        for start in range(0, len(list2_IDs), batch_size):
            X_batch = []
            y_batch = []

            end = min(start + batch_size, len(list2_IDs))
            index = list(indexes[start:end]) 
            # Get Input Batches     
            current_batch_size = len(indexes[start:end])
           
            X, y= mydata_generation(index,current_batch_size,dim,dim2,n_channels,list_IDs,labels,list_demIDs,n_classes) 
        
            yield X, y # Return batch input and output
                
@threadsafe_generator
def validation_generator(list_IDs, list_demIDs,labels, batch_size=10, dim=(182,218,182), dim2=(3,), n_channels=1,
                    n_classes=5, shuffle=False):
    while True:
        list2_IDs=list_IDs[0] # control the batch size
        indexes = np.arange(len(list2_IDs))
        if shuffle == True:
            np.random.shuffle(indexes)
            
        for start in range(0, len(list2_IDs), batch_size):
            X_batch = []
            y_batch = []

            end = min(start + batch_size, len(list2_IDs))
            index = list(indexes[start:end]) 
            # Get Input Batches   
            current_batch_size = len(indexes[start:end])
            
            X, y= mydata_generation(index,current_batch_size,dim,dim2,n_channels,list_IDs,labels,list_demIDs,n_classes)  
        
            yield X, y # Return batch input and output

@threadsafe_generator
def prediction_generator(list_IDs, list_demIDs,labels, batch_size=10, dim=(182,218,182), dim2=(3,), n_channels=1,
                    n_classes=5, shuffle=False):
    while True:
        list2_IDs=list_IDs[0] # control the batch size
        indexes = np.arange(len(list2_IDs))
        
        if shuffle == True:
            np.random.shuffle(indexes)
            
        for start in range(0, len(list2_IDs), batch_size):
            X_batch = []
            y_batch = []

            end = min(start + batch_size, len(list2_IDs))
            index = list(indexes[start:end]) 
            # Get Input Batches
            current_batch_size = len(indexes[start:end])            
            X, y= mydata_generation(index,current_batch_size,dim,dim2,n_channels,list_IDs,labels,list_demIDs,n_classes) 
            yield X, y # Return batch input and output
                
                
def mydata_generation(indexes,batch_size,dim,dim2,n_channels,list_IDs,labels,list_demIDs,n_classes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization 
        X1 = np.empty((batch_size, *dim, n_channels)) # channel       
        X2 = np.empty((batch_size, *dim2)) # Demographic data
        y = np.empty((batch_size)) # output
        X= [] # batches list 
        for rk in range (0,len(list_IDs)):            
            # Find list of IDs
            list_IDs_temp = [list_IDs[rk][k] for k in indexes]             
            i=0
            nk=0
            for counter in list_IDs_temp:
                # Read Nifti mage 
                file_name=counter 
                example_nii = os.path.join(data_path, file_name)
                imageobj = nib.load(example_nii) # Read .nii image          
                image_ch= imageobj.get_fdata()             
                (z1,y1,x1) =image_ch.shape # Find Dimension                
                image_ch= image_ch.reshape((int(z1), int(y1), int(x1),int(n_channels)))                
                X1[i,:,:,:,:]=image_ch                 
                if rk==0:
                    indrk = list_IDs[rk].index(file_name)
                    ytrain= labels[indrk]
                    dem_data=np.asarray(list_demIDs[indrk])
                    dem_data = dem_data.reshape((-1))
                    X2[nk,:]=dem_data # demographic data
                    y[nk]=ytrain #output  
                    nk=nk+1
                i=i+1
            X.append(X1)  # add batch file of a channel
        X.append(X2)  # add demographic data
       
        return X, keras.utils.to_categorical(y, n_classes)
     
    