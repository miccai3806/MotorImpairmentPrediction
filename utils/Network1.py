# -*- coding: utf-8 -*-
"""
Created on Monday Feb  17 13:00:00 2025
@author: Dr. xxxx
"""
# Base libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard 
from tensorflow.keras.utils import get_file
from tensorflow.keras.layers import Conv3D, MaxPooling3D, AveragePooling3D, GlobalMaxPooling3D,ZeroPadding3D,GlobalAveragePooling3D,Dropout
from tensorflow.keras.layers import Lambda,Reshape, Multiply, Add,add, Permute,concatenate,BatchNormalization,Activation,Flatten,Input,Dense
from tensorflow.keras.optimizers import Adam,SGD,RMSprop,Adagrad,Adadelta, Adamax
from itertools import product  

from tensorflow.keras import regularizers, backend as K  
import numpy as np
from tensorflow.keras.initializers import he_normal, random_normal

################# MODELS ######################################################
#### CBAM Components  #######################
def channel_attention(input_feature, w_l2, ratio=8):
    channel = input_feature.shape[-1]

    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_regularizer=regularizers.l2(w_l2))
    shared_layer_two = Dense(channel, kernel_regularizer=regularizers.l2(w_l2))

    avg_pool = GlobalAveragePooling3D()(input_feature)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling3D()(input_feature)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = tf.keras.layers.Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)

    return Multiply()([input_feature, cbam_feature])

def spatial_attention(input_feature):
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    max_pool = Lambda(lambda x: tf.reduce_max(x, axis=-1, keepdims=True))(input_feature)
    concat = tf.keras.layers.Concatenate(axis=-1)([avg_pool, max_pool])

    cbam_feature = Conv3D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid')(concat)

    return Multiply()([input_feature, cbam_feature])

def cbam_block(input_feature, w_l2, ratio=8):
    ca_feature = channel_attention(input_feature, w_l2, ratio)
    sa_feature = spatial_attention(ca_feature)
    return sa_feature

# SBAM Components
def sbam_block(input_feature, w_l2):
    ca_feature = channel_attention(input_feature, w_l2)
    avg_pool = Lambda(lambda x: tf.reduce_mean(x, axis=-1, keepdims=True))(input_feature)
    sbam_feature = Conv3D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid')(avg_pool)
    
    return Multiply()([ca_feature, sbam_feature])
###############################################################################

def Block1 (input, kernel_size,name,w_l2):
    
  output = Conv3D(kernel_size, (3, 3, 3), strides=(1, 1, 1),kernel_regularizer=regularizers.l2(w_l2),
                  bias_regularizer=regularizers.l2(w_l2),padding="same", name=name+'_conv1')(input)
  output=BatchNormalization(axis=3)(output)  

  output=Activation('relu')(output) 
  return output

def Block2 (input, kernel_size,name,w_l2):
  x=input  
  # first layer  
  output = Block1(input, kernel_size,name,w_l2)
  
  # second layer
  output = Conv3D(kernel_size, (3, 3, 3), strides=(1, 1, 1),kernel_regularizer=regularizers.l2(w_l2),
                  bias_regularizer=regularizers.l2(w_l2),padding="same", name=name+'_conv2')(input)
  output=BatchNormalization(axis=3)(output)

  # merge
  output=Add()([output, x])
  
  output=Activation('relu')(output) 

  return output 
  
from tensorflow.python.ops.gen_array_ops import shape
def Block3 (input,kernel_size,name,w_l2,use_cbam=False,use_sbam=False):
  x=input  
  # first layer  
  output = Block1(input, kernel_size,name,w_l2)
  
  # second layer
  output = Conv3D(kernel_size, (3, 3, 3), strides=(1, 1, 1),kernel_regularizer=regularizers.l2(w_l2),
                  bias_regularizer=regularizers.l2(w_l2),padding="same", name=name+'_conv2')(input)
  output=BatchNormalization(axis=3)(output)
  output=AveragePooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same",name=name+"avgpool1")(output)  
  
  # residual-layer
  x = Conv3D(kernel_size, (1, 1, 1), strides=(2, 2, 2),kernel_regularizer=regularizers.l2(w_l2),
                  bias_regularizer=regularizers.l2(w_l2),padding="same", name=name+'_resconv1')(x)
  x=BatchNormalization(axis=3)(x)
 
  if use_cbam:
      output = cbam_block(output, w_l2)
  if use_sbam:
      output = sbam_block(output, w_l2)    
  
  # merge 
  output=Add()([output, x])
  output=Activation('relu')(output) 
  
  return output
  
def Block4 (input, kernel_size,name,w_l2,use_cbam=False,use_sbam=False):    
  # first layer  
  output = Block1(input, kernel_size,name,w_l2)
  # second layer
  output = Conv3D(kernel_size, (3, 3, 3), strides=(1, 1, 1),kernel_regularizer=regularizers.l2(w_l2),
                  bias_regularizer=regularizers.l2(w_l2),padding="same", name=name+'_conv2')(input)
  output=BatchNormalization(axis=3)(output)
  
  if use_cbam:
      output = cbam_block(output, w_l2)
  if use_sbam:
      output = sbam_block(output, w_l2)
  
  output=GlobalAveragePooling3D(name=name+'_globalavpool')(output)
  
  return output 
  
###############################################################################
def get_model(input,layers,batchnor_bool,name,w_l2):
    ## 1st convolution
    model = Conv3D(layers[0], (3, 3, 3), strides=(1, 1, 1),kernel_regularizer=regularizers.l2(w_l2),
                   bias_regularizer=regularizers.l2(w_l2), data_format='channels_last',padding="same", 
                   name=name+'_conv1')(input)
    model=BatchNormalization(axis=3)(model)
    model=Activation('relu')(model) 
    model = AveragePooling3D(pool_size=(3, 3, 3), strides=(2, 2, 2), padding="same", name=name+'pool1')(model)
   
    
    # Stage1
    model=Block1 (model, layers[1],name+'_block12',w_l2)
    
    # Stage2
    model=Block2 (model, layers[2],name+'_block21',w_l2)
    model=Block2 (model, layers[3],name+'_block22',w_l2)
    model=Block2 (model, layers[4],name+'_block23',w_l2)
    model=Block2 (model, layers[5],name+'_block24',w_l2)
    model=Block2 (model, layers[6],name+'_block25',w_l2)
    
    # Stage3
    model=Block3 (model, layers[7],name+'_block31',w_l2,use_cbam=True, use_sbam=False)
    model=Block3 (model, layers[8],name+'_block32',w_l2,use_cbam=True, use_sbam=False)
    model=Block3 (model, layers[9],name+'_block33',w_l2,use_cbam=True, use_sbam=False)
    model=Block3 (model, layers[10],name+'_block34',w_l2,use_cbam=True, use_sbam=False)

    # Stage4
    model=Block4(model, layers[11],name+'_block41',w_l2,use_cbam=True, use_sbam=False)

    return model

# multi-channels 
def get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers,input_shape, input_dem, drop_bool,drop_out,num_classes,inputs_num,batchnor_bool,out_tf):
    w_l2=1e-5 

    if optimization_option==1: # SGD
        optimizer= SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
    elif optimization_option==2: #RMSprop
        optimizer=RMSprop(learning_rate=lr, rho=0.9)
    elif optimization_option==3: # Adam
        optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
    elif optimization_option==4: # AdaGrad
        optimizer=Adagrad(learning_rate=lr,initial_accumulator_value=0.1,epsilon=1e-07)  
    
    total=[]
    inputs=[]
    for i in range(0,inputs_num):
        # DTI maps /MRI images 
        input1= Input(shape=input_shape)        
        model1=get_model(input1,layers,batchnor_bool,name='input'+str(i),w_l2=w_l2) 
        total.append(model1)
        inputs.append(input1)

    # Demographic inputs    
    input2=Input(shape=input_dem)    
    model_deg=(Dense(1, activation='relu', kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2), name='dense_dem'))(input2) # input_shape=(1,)
    inputs.append(input2)
      
    # merge inputs
    total.append(model_deg)
    model=concatenate(total)
    model=BatchNormalization(name='feature_bn')(model)
    
    # FC layers
    model = Dense(layers[12], activation='relu', kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2), name='dense1')(model) # fc1
    if drop_bool==1:
        model = Dropout(drop_out)(model) 
    model=BatchNormalization()(model)
    model = Dense(layers[13], activation='relu', kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2), name='dense2')(model) #fc2
    if drop_bool==1:
        model = Dropout(drop_out)(model)
    model=BatchNormalization()(model)
#    
    # Output Layer             
    output = Dense(num_classes, activation=out_tf, kernel_regularizer=regularizers.l2(w_l2), bias_regularizer=regularizers.l2(w_l2), name='dense3')(model)
    
    model = Model(inputs=inputs, outputs=output) 
     
    # Compile    
    model.compile(loss=loss, optimizer=optimizer, metrics=[metric]) 
    
    return model