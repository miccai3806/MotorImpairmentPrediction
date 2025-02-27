"""
Created on 2025
@author: Dr. XXXXX
"""

import tensorflow as tf
from  tensorflow import keras

def get_session(session_option):
    # Open GPU session for Tensorflow 2.0 
    if session_option==1: # TF version<2.0
        config = tf.compat.v1.ConfigProto()# device_count = {'GPU': 0,'CPU':8})     
        config.gpu_options.allow_growth = True
        graph = tf.compat.v1.get_default_graph()
        sess = tf.compat.v1.Session(graph=graph, config=config)
        tf.compat.v1.keras.backend.set_session(sess)
    elif session_option==0: # TF version>2.0       
        config = tf.ConfigProto(device_count={'GPU':0, 'CPU':8}) # max: 1 gpu, 56 cpu
        sess = tf.Session(config=config)
        keras.backend.set_session(sess)
    return sess
  
def close_session(session_option):
    if session_option==1 : # TF version>2.0
        sess = tf.compat.v1.keras.backend.get_session()
        tf.compat.v1.keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()
        sess.close()
        del sess
    elif session_option==0 : # TF version<2.0
        sess = get_session(session_option)
        tf.reset_default_graph()
        sess.close()
        del sess
