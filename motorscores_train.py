# -*- coding: utf-8 -*-
'''
-The Motor Scores Classification with DTI and MR images of Stroke Patients
-Multimodal DL model with Age, Gender and Stroke Time
based Keras and Tensorflow
Created on Monday Feb  17 13:00:00 2025
@author: Dr. XXXXX
'''
for i in range(1, 11):  # CV1-CV10
    dataset_cv = [f"CV{i}"]  # 
    path_file = fr"C:\Users\USER\Desktop\MotorScoresClassification\CV{i}" 
    modelpath=r"C:\Users\USER\Desktop\MotorScoresClassification"
    datasetpath=r"C:\Users\USER\Desktop\MotorScores_DL\Dataset_2mm\Dataset_CrossValidation"

    import sys 
    sys.path.append(r"C:\Users\USER\Desktop\MotorScoresClassification\utils")
    from data_handling1 import *
    from Network1 import *
    from metric_utils1 import *
    from tf_session1 import *
    from Data_Generator1 import *
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint,LearningRateScheduler,TensorBoard 

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
    cc=10 # Number of training repeat of each subdataset (CV)
    num_classes = 2 
    batch_size = 8
    epochs=100     
    optimization_option=3 # 1-SGD, 2-RMSProp, 3-Adam    
    metric="categorical_accuracy" 
    loss="categorical_crossentropy" #"binary_crossentropy"     
    layers=[32,16,16,16,16,16,16,16,64,128,256,512,256,256] # filter_size
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
        
        # Adjust class weights
        from sklearn.utils import class_weight
        class_weight = class_weight.compute_class_weight('balanced'
                                                       ,classes=np.unique(ytraining_list)
                                                       ,y=ytraining_list)
        class_weight_dict = dict(zip(np.unique(ytraining_list), class_weight))
        
        #--------- Create Data-Generators----------------------------------------------
        tr_generator = train_generator(training_listAll, training_demlist, ytraining_list, batch_size,input_shape,input_dem,
                                          n_channels,num_classes,shuffle) 
             
        step_size_train =  (np.ceil(len(training_listAll[0])/ float(batch_size))).astype(np.int32)
        
        if valbool==1:
            val_generator =validation_generator(validation_listAll, validation_demlist, yvalidation_list, batch_size,input_shape,
                                                input_dem, n_channels,num_classes,shuffle)
            step_size_val =(np.ceil(len(validation_listAll[0])/float(batch_size))).astype(np.int32)           
        if testrate>0.0:
            test_generator =prediction_generator(prediction_listAll, prediction_demlist, yprediction_list, batch_size,input_shape,
                                                 input_dem, n_channels,num_classes,shuffle)
            step_size_test=(np.ceil(len(prediction_listAll[0])/float(batch_size))).astype(np.int32)  
            
        #------------------------------------------------------------------------------   
       
        #-----------Repeat of Training -----------------------------------------------------------         
        for ii in range(cc):    
        # construct CNN structure             
            model=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
            model.summary()
        #----------------TRAIN NETWORK-------------------------------------------------
            # # Stopping criteria   
            weights_file = path_file+'\MotorScore_weights'+cvpath+str(ii)+'.hdf5' 
            callbacks = [
                    EarlyStopping(monitor='val_categorical_accuracy',
                                    patience=50,
                                    verbose=1,
                                    min_delta=1e-5,
                                    restore_best_weights=True,
                                    mode='max'),
                    ReduceLROnPlateau(
                        monitor='val_loss',
                        factor=0.5,
                        patience=20,  
                        verbose=1,
                        min_delta=1e-4,
                        mode='min',   
                        min_lr=5e-6   
                    ),               
                     ModelCheckpoint(monitor='val_categorical_accuracy', 
                                     filepath=weights_file,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max',
                                     period=1)]    
            print("TRAINING STARTED------------------------------------------------")
            if valbool==1: 
                history = model.fit(tr_generator,validation_data=val_generator, 
                                                  validation_steps=step_size_val, steps_per_epoch=step_size_train, 
                                                  epochs=epochs, 
                                                  callbacks=callbacks, 
                                                  max_queue_size = 1, 
                                                  workers = 0,
                                                  use_multiprocessing=False, 
                                                  shuffle=shuffle,
                                                  class_weight=class_weight_dict) 
                #Save History Result
                h_dict = {'history': history.history}
                savemat(path_file+"\\history"+cvpath+str(ii)+"1.mat", h_dict)    
                    
        #-------- SAVE MODEL------------------------------------------------------------
            fnm=path_file+"\\model"+cvpath+str(ii)+".json"
            model_json = model.to_json()
            with open(fnm, "w") as json_file:
                json_file.write(model_json)
            # serialize weights to HDF5
            fnm1=path_file+"\\result"+cvpath+str(ii)+".h5"
            model.save_weights(fnm1)
        
            print("Saved model to disk")
            print("TRAINING FINISHED-----------------------------------------------")    
        # ---------------------------------------------------------------------------
        # FIRST EVALUATION
            print("EVALUATION STARTED----------------------------------------------")    
            train_score = model.evaluate(tr_generator, steps=step_size_train, verbose=1,max_queue_size=1, workers=0, use_multiprocessing=False)
            print("TRANINING EVALUATION FINISHED")
            val_score=[]
            test_score=[]
            if valbool==1:
                val_score = model.evaluate(val_generator, steps=step_size_val, verbose=1, max_queue_size=1, workers=0, use_multiprocessing=False)
                print("VALIDATION EVALUATION FINISHED")
            if testrate>0.0:
                test_score = model.evaluate(test_generator, steps=step_size_test, verbose=1, max_queue_size=1, workers=0,use_multiprocessing=False)   
            
            # FIND test output
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_pred= model.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_pred=test_pred[:len(y_test),:]
            test_pred2=np.argmax(test_pred,axis=1)
            test_pred2 = np.array(test_pred2, 'float32')
            test_pred3 = keras.utils.to_categorical(test_pred2, num_classes)   
            test_result=np.zeros((num_classes, 13)) # 
            test_result_mean,test_result,confusion=metric_all(path_file,cvpath,ii,'0',train_score, test_score, valbool,val_score,history,num_classes,y_test,test_pred3)       
            
            if ii==0:
                all_result=test_result 
                all_confusion=confusion
                all_meanresult=test_result_mean
            else:
                all_result=np.append(all_result,test_result,axis=0)     
                all_confusion=np.append(all_confusion,confusion,axis=0) 
                all_meanresult=np.append(all_meanresult,test_result_mean,axis=0)
            
            print("PREDICTION EVALUATION FINISHED---------------------------------------------")
            
            # save test results
            r_dict = {
                      'ytest': y_test,
                      'pred_test':test_pred3,
                      'confusion':confusion,
                      'test_result':test_result,
                      'prediction_list':prediction_listAll[0] # BURAYI KONTROL ET MUTLAKA....
                      }
        
            savemat(path_file+"\\testresult"+cvpath+str(ii)+"1.mat", r_dict)
        
         
        #--- SECOND EVALUATION WITH BEST WEIGHTS ---------------------------------------------------------- 
            # load best model weights   
            model_selection=1
            if model_selection==1:
                drop_bool=0
                batch_bool=0
                model=get_MultiModalNetwork3D(optimization_option, metric, loss, lr, layers, input_size,input_dem, drop_bool, drop_out,num_classes,1,batchnor_bool=1,out_tf='softmax')  
                model.load_weights(weights_file)
        
            print("EVALUATION STARTED----------------------------------------------")    
            train_score = model.evaluate(tr_generator, steps=step_size_train, verbose=1,max_queue_size=1, workers=0, use_multiprocessing=False)
            print("TRANINING EVALUATION FINISHED")
            val_score=[]
            test_score=[]
            if valbool==1:
                val_score = model.evaluate(val_generator, steps=step_size_val, verbose=1, max_queue_size=1, workers=0, use_multiprocessing=False)
                print("VALIDATION EVALUATION FINISHED")
            if testrate>0.0:
                test_score = model.evaluate(test_generator, steps=step_size_test, verbose=1, max_queue_size=1, workers=0,use_multiprocessing=False)   
            
            # FIND test output
            y_test=keras.utils.to_categorical(yprediction_list, num_classes)    
            test_pred= model.predict(test_generator,steps=step_size_test,verbose=1,workers=0,max_queue_size=1, use_multiprocessing=False)
            test_pred=test_pred[:len(y_test),:]
            test_pred2=np.argmax(test_pred,axis=1)
            test_pred2 = np.array(test_pred2, 'float32')
            test_pred3 = keras.utils.to_categorical(test_pred2, num_classes)   
            test_result=np.zeros((num_classes, 13)) # 
            test_result_mean,test_result,confusion=metric_all(path_file,cvpath,ii,'1',train_score, test_score, valbool,val_score,history,num_classes,y_test,test_pred3)       
            print("PREDICTION EVALUATION FINISHED---------------------------------------------")
            
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
        
        ### SAVE RESULT
        r_dict = {
                      'all_result': all_result,
                      'all_confusion':all_confusion,
                      'all_meanresult':all_meanresult
                  }
        savemat(path_file+"\\"+cvpath+"_ALL.mat", r_dict)

        #-------- Delete Model--------------------------------------------------------
        del model  
        # =============================================================================
        #-----------CLOSE GPU-SESSION--------------------------------------------------
        close_session(session_option)
        #------------------------------------------------------------------------------
