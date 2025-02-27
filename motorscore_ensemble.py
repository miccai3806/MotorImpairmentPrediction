# -*- coding: utf-8 -*-
'''
-The Motor Scores Classification with DTI and MR images of Stroke Patients
-Multimodal DL model with Age, Gender and Stroke Time
based Keras and Tensorflow
Created on Monday Feb  17 13:00:00 2025
@author: Dr. XXXXX
'''
for i in range(1, 11):  # CV1-CV10
    dataset_cv = [f"CV{i}"]  # CV klasörünü güncelle
    print(f"Prediction is being performed for {dataset_cv[0]}")    
    path_file = fr"C:\Users\USER\Desktop\MotorScoresClassification\CV{i}"  # Path'i güncelle
    modelpath=r"C:\Users\USER\Desktop\MotorScoresClassification"
    datasetpath=r"C:\Users\USER\Desktop\MotorScores_DL\Dataset_2mm\Dataset_CrossValidation"
    ensemblemodelpath=r"C:\Users\USER\Desktop\MotorScoresClassification\Ensemble_Models"
    
    # import folders
    import sys 
    sys.path.append(r"C:\Users\USER\Desktop\MotorScoresClassification\utils")
    from data_handling1 import *
    from Network1 import *
    from metric_utils2 import *
    from tf_session1 import *
    from Data_Generator1 import *
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,LearningRateScheduler,TensorBoard # Erken durdurma için

    import multiprocessing
    import math
    from scipy.io import savemat
    
    # load model
    from tensorflow.keras.models import Sequential, Model, load_model
    from tensorflow.keras.models import model_from_json
    import json
    
    # Open GPU or CPU session
    session_option=1 # 0: TF<2.0, #1: TF>2.0
    get_session(session_option)
    
    # Compile and Network options
    cc=1 # Cross correlation
    num_classes = 2 # Outputs: Good/Poor
    batch_size = 8
    epochs=100 # 
    tf_bool=0  # 1:Transfer Learning    
    tf_type=1  
    optimization_option=3 # 1-SGD, 2-RMSProp, 3-Adam    
    metric="categorical_accuracy" 
    loss="categorical_crossentropy"    
    layers=[32,16,16,16,16,16,16,16,64,128,256,512,256,256] # model's filter_size
    drop_bool=1
    drop_out=0.3
    shuffle=False # to mix lists
    batchnor_bool=1
    out_tf='softmax'
    
    
    # --------Prepare trainset, prediction and validation sets--------------------
    x_train, y_train, x_test, y_test, x_val, y_val= [], [], [], [], [], []
    strokeclasses = ["0", "1"] 
    imageclasses=["GM"]  # "AD", "FA", "GM", "MD", "RD", "WM"
    
    demographic_bool=1 # 0:no, 1:yes
    input_dem=(3,) # age, gender and stroke time
    n_channels=1 # 1:gray, 3:rgb
    
    number_image=91 # dimension-0
    height=109  # dimension-1
    width=91   # dimension-2
    
    input_real=(number_image, height, width) # Images size
    input_shape=(number_image, height, width) # Images size
    input_size=(number_image, height, width, n_channels)  
    
    valbool=1     # Validation set control
    trainrate=1.0 # Training rate
    valrate=0.1  # Validation rate
    testrate=0.2  # Testing rate 
    
    for cvpath in dataset_cv:
        lr=0.0001
        training_demlist,prediction_demlist,validation_demlist=[],[],[]
        ytraining_list, yprediction_list, yvalidation_list=[],[],[] 
        training_listAll,prediction_listAll,validation_listAll=[],[],[]
        
        #-----------CROSS CORRELATION-----------------------------------------------------------         
        for ii in range(cc):    
        # construct CNN structure             
            seed(30)
            drop_bool=0
            batch_bool=0  
            
            ### WM  model ####################################################
            modelWM=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
            imageclasses=["WM"]
            
            training_listAll, training_demlist, ytraining_list,validation_listAll,validation_demlist, yvalidation_list,prediction_listAll,prediction_demlist, yprediction_list=ensemble_data(datasetpath,cvpath,num_classes,strokeclasses,valbool,trainrate,testrate,valrate,imageclasses)
            tr_generator,  step_size_train,  val_generator, step_size_val, test_generator, step_size_test=ensemble_datagenerator(training_listAll, training_demlist, ytraining_list,
                                       validation_listAll, validation_demlist, yvalidation_list,
                                       prediction_listAll, prediction_demlist, yprediction_list,
                                       batch_size,input_shape,input_dem,n_channels,num_classes,shuffle)           
            
            weights_file = ensemblemodelpath+'\WM_MotorScore_weights.hdf5' # 
            modelWM.load_weights(weights_file)
            print ("WM model loaded")
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_predWM= modelWM.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_predWM=test_predWM[:len(y_test),:] 
            print ("WM model evaluation finished")

            ### GM model ######################################################
            modelGM=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
            imageclasses=["GM"]
            training_listAll, training_demlist, ytraining_list,validation_listAll,validation_demlist, yvalidation_list,prediction_listAll,prediction_demlist, yprediction_list=ensemble_data(datasetpath,cvpath,num_classes,strokeclasses,valbool,trainrate,testrate,valrate,imageclasses)
            tr_generator,  step_size_train,  val_generator, step_size_val, test_generator, step_size_test=ensemble_datagenerator(training_listAll, training_demlist, ytraining_list,
                                       validation_listAll, validation_demlist, yvalidation_list,
                                       prediction_listAll, prediction_demlist, yprediction_list,
                                       batch_size,input_shape,input_dem,n_channels,num_classes,shuffle)           
            
            weights_file = ensemblemodelpath+'\GM_MotorScore_weights.hdf5' # 
            modelGM.load_weights(weights_file)
            print ("GM model loaded")
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_predGM= modelGM.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_predGM=test_predGM[:len(y_test),:]
            print ("GM model evaluation finished")
   
            ### FA model ######################################################
            modelFA=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
            imageclasses=["FA"]
            training_listAll, training_demlist, ytraining_list,validation_listAll,validation_demlist, yvalidation_list,prediction_listAll,prediction_demlist, yprediction_list=ensemble_data(datasetpath,cvpath,num_classes,strokeclasses,valbool,trainrate,testrate,valrate,imageclasses)
            tr_generator,  step_size_train,  val_generator, step_size_val, test_generator, step_size_test=ensemble_datagenerator(training_listAll, training_demlist, ytraining_list,
                                       validation_listAll, validation_demlist, yvalidation_list,
                                       prediction_listAll, prediction_demlist, yprediction_list,
                                       batch_size,input_shape,input_dem,n_channels,num_classes,shuffle)           
            
            weights_file = ensemblemodelpath+'\FA_MotorScore_weights.hdf5' # 
            modelFA.load_weights(weights_file)
            print ("FA model loaded")
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_predFA= modelFA.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_predFA=test_predFA[:len(y_test),:]
            print ("FA model evaluation finished")

            ### AD model ######################################################
            modelAD=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
            imageclasses=["AD"]
            training_listAll, training_demlist, ytraining_list,validation_listAll,validation_demlist, yvalidation_list,prediction_listAll,prediction_demlist, yprediction_list=ensemble_data(datasetpath,cvpath,num_classes,strokeclasses,valbool,trainrate,testrate,valrate,imageclasses)
            tr_generator,  step_size_train,  val_generator, step_size_val, test_generator, step_size_test=ensemble_datagenerator(training_listAll, training_demlist, ytraining_list,
                                       validation_listAll, validation_demlist, yvalidation_list,
                                       prediction_listAll, prediction_demlist, yprediction_list,
                                       batch_size,input_shape,input_dem,n_channels,num_classes,shuffle)           
            
            weights_file = ensemblemodelpath+'\AD_MotorScore_weights.hdf5' # 
            modelAD.load_weights(weights_file)
            print ("AD model loaded")
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_predAD= modelAD.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_predAD=test_predAD[:len(y_test),:]
            print ("AD model evaluation finished")

            ### MD model ######################################################
            modelMD=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
            imageclasses=["MD"]
            training_listAll, training_demlist, ytraining_list,validation_listAll,validation_demlist, yvalidation_list,prediction_listAll,prediction_demlist, yprediction_list=ensemble_data(datasetpath,cvpath,num_classes,strokeclasses,valbool,trainrate,testrate,valrate,imageclasses)
            tr_generator,  step_size_train,  val_generator, step_size_val, test_generator, step_size_test=ensemble_datagenerator(training_listAll, training_demlist, ytraining_list,
                                       validation_listAll, validation_demlist, yvalidation_list,
                                       prediction_listAll, prediction_demlist, yprediction_list,
                                       batch_size,input_shape,input_dem,n_channels,num_classes,shuffle)           
            
            weights_file = ensemblemodelpath+'\MD_MotorScore_weights.hdf5' # 
            modelMD.load_weights(weights_file)
            print ("MD model loaded")
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_predMD= modelMD.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_predMD=test_predMD[:len(y_test),:]
            print ("MD model evaluation finished")

            ### RD model ######################################################
            modelRD=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
            imageclasses=["RD"]
            training_listAll, training_demlist, ytraining_list,validation_listAll,validation_demlist, yvalidation_list,prediction_listAll,prediction_demlist, yprediction_list=ensemble_data(datasetpath,cvpath,num_classes,strokeclasses,valbool,trainrate,testrate,valrate,imageclasses)
            tr_generator,  step_size_train,  val_generator, step_size_val, test_generator, step_size_test=ensemble_datagenerator(training_listAll, training_demlist, ytraining_list,
                                       validation_listAll, validation_demlist, yvalidation_list,
                                       prediction_listAll, prediction_demlist, yprediction_list,
                                       batch_size,input_shape,input_dem,n_channels,num_classes,shuffle)           
            
            weights_file = ensemblemodelpath+'\RD_MotorScore_weights.hdf5' # 
            modelRD.load_weights(weights_file)
            print ("RD model loaded")
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_predRD= modelRD.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_predRD=test_predRD[:len(y_test),:]
            print ("RD model evaluation finished")
         
            import numpy as np

            ensemble_prediction = np.mean(
                [test_predWM, test_predFA],#test_predWM, test_predGM, test_predFA, test_predAD, test_predMD, test_predRD
                axis=0) 

            test_pred2=np.argmax(ensemble_prediction,axis=1)
            test_pred2 = np.array(test_pred2, 'float32')
            test_pred3 = keras.utils.to_categorical(test_pred2, num_classes)   
            test_result=np.zeros((num_classes, 13)) # 
            train_score=0,
            test_score=0,
            val_score=0
            history=[]
            
            test_result_mean,test_result,confusion=metric_all(path_file,cvpath,ii,'1',train_score, test_score, valbool,val_score,history,num_classes,y_test,test_pred3)       

            print("PREDICTION EVALUATION FINISHED---------------------------------------------")
            
            if ii==0:
                all_result=test_result 
                all_confusion=confusion
                all_meanresult=test_result_mean
            else:
                all_result=np.append(all_result,test_result,axis=0)     
                all_confusion=np.append(all_confusion,confusion,axis=0) 
                all_meanresult=np.append(all_meanresult,test_result_mean,axis=0)
            
            # save test results
            r_dict = {
                      'ytest': y_test,
                      'pred_test':test_pred3,
                      'confusion':confusion,
                      'test_result':test_result,
                      'prediction_list':prediction_listAll[0]
                      }
            savemat(path_file+"\\testresult"+cvpath+str(ii)+"2.mat", r_dict)
        #------------------------------------------------------------------------- 
        
        # ### SAVE RESULT
        r_dict = {
                      'all_result': all_result,
                      'all_confusion':all_confusion,
                      'all_meanresult':all_meanresult
                  }
        savemat(path_file+"\\"+cvpath+"_ALL.mat", r_dict)


        # #-------- Delete Model--------------------------------------------------------
        # del model  
        # =============================================================================
        #-----------CLOSE GPU-SESSION--------------------------------------------------
        close_session(session_option)
        #------------------------------------------------------------------------------
