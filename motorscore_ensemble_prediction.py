# -*- coding: utf-8 -*-
'''
-The Motor Scores Classification with DTI and MR images of Stroke Patients
-Multimodal DL model with Age, Gender and Stroke Time
based Keras and Tensorflow
Created on Monday Feb  17 13:00:00 2025
@author: Dr. XXXXX
'''  
# import folders
import sys 
sys.path.append(r"C:\Users\USER\Desktop\MotorScoresClassification\utils")
from data_handling1 import *
from Network1 import *
from metric_utils2 import *
from tf_session1 import *
from Data_Generator1 import *
    
# load model
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.models import model_from_json
import json

def read_niftifiles(file_name):    
    example_nii = os.path.join(data_path, file_name)
    imageobj = nib.load(example_nii) # Read .nii image          
    image_ch= imageobj.get_fdata()             
    (z1,y1,x1) =image_ch.shape # Find Dimension                
    image_ch= image_ch.reshape((int(z1), int(y1), int(x1),1))    
    return image_ch

def prepare_batch(file_path,demographic_data):
    X1 = np.zeros((1, input_size[0],input_size[1],input_size[2],input_size[3])) # image batch 
    X2 = np.zeros((1, input_dem[0])) # Demographic data  
    X=[] 
    
    image=read_niftifiles(file_path)
    X1[0,:,:,:,:]=image  
    X2[0,:]=demographic_data 
    X.append(X1)
    X.append(X2) 
    return X

input_dem=(3,) # age, gender and stroke time
n_channels=1 # 1:gray, 3:rgb    
number_image=91 # dimension-0
height=109  # dimension-1
width=91   # dimension-2    
input_real=(91, 109, 91) # Images size
input_shape=(number_image, height, width) # Images size
input_size=(number_image, height, width, n_channels)  

lr=0.0001
drop_bool=0
batch_bool=0
shuffle=False 
batchnor_bool=1
num_classes = 2 # Outputs: Good/Poor
optimizer=Adam(learning_rate=lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
 
######## Load  model file #####################################################
model_json_path = r"C:\Users\USER\Desktop\MotorScoresClassification\Ensemble_Models\model.json"
# Load model architecture from JSON file
with open(model_json_path, "r") as json_file:
    model_json = json_file.read() 
###############################################################################

ensemblemodelpath=r"C:\Users\USER\Desktop\MotorScoresClassification\Ensemble_Models"

########### Define demographic data, image files path #########################
age_max=85
stroke_time_max=6368 # time after stroke: converted to days

test_selection=0 # 0: healty/stroke with good motor-impairment data, 1: stroke with poor motor-impairment data

if test_selection==0: 
    demographic_data = [31, 0, 0] # age, gender,stroke_time
    demographic_data_norm=[31/age_max,0,0/stroke_time_max] # !!!!
    wm_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\t1_c4_structural_WM.nii.gz'
    gm_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\t1_c4_structural_GM.nii.gz'
    ad_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_c4_dti2_AD_reg.nii.gz'
    fa_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_c4_dti2_FA_reg.nii.gz'
    md_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_c4_dti2_MD_reg.nii.gz'
    rd_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_c4_dti2_RD_reg.nii.gz'
else: 
    demographic_data = [45, 0, 1568] # age, gender,stroke_time
    demographic_data_norm=[45/age_max,0,1568/stroke_time_max] # !!!!
    wm_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\t1_p4_structural_WM.nii.gz'
    gm_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\t1_p4_structural_GM.nii.gz'
    ad_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_p4_dti2_AD_reg.nii.gz'
    fa_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_p4_dti2_FA_reg.nii.gz'
    md_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_p4_dti2_MD_reg.nii.gz'
    rd_file=r'C:\Users\USER\Desktop\MotorScoresClassification\Test_Data\n1_p4_dti2_RD_reg.nii.gz'
         
### WM  model ####################################################
modelWM=model_from_json(model_json) 
weights_file = ensemblemodelpath+'\WM_MotorScore_weights.hdf5' # 
modelWM.load_weights(weights_file)

modelWM.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
print ("WM model loaded")

X=prepare_batch(wm_file,demographic_data_norm)

test_predWM= modelWM.predict(X)
print ("WM model evaluation finished")

### GM model ######################################################
modelGM= model_from_json(model_json)    
weights_file = ensemblemodelpath+'\GM_MotorScore_weights.hdf5' # 
modelGM.load_weights(weights_file)

modelWM.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
print ("GM model loaded")

X=prepare_batch(gm_file,demographic_data_norm)
 
test_predGM= modelGM.predict(X)
print ("GM model evaluation finished")
   
### FA model ######################################################
modelFA=model_from_json(model_json)
weights_file = ensemblemodelpath+'\FA_MotorScore_weights.hdf5' # 
modelFA.load_weights(weights_file)

modelFA.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
print ("FA model loaded")

X=prepare_batch(fa_file,demographic_data_norm)
  
test_predFA= modelFA.predict(X)
print ("FA model evaluation finished")

### AD model ######################################################
modelAD=model_from_json(model_json)
weights_file = ensemblemodelpath+'\AD_MotorScore_weights.hdf5' # 
modelAD.load_weights(weights_file)

modelAD.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
print ("AD model loaded")

X=prepare_batch(ad_file,demographic_data_norm)
   
test_predAD= modelAD.predict(X)
print ("AD model evaluation finished")

### MD model ######################################################
modelMD=model_from_json(model_json)       
weights_file = ensemblemodelpath+'\MD_MotorScore_weights.hdf5' # 
modelMD.load_weights(weights_file)

modelMD.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
print ("AD model loaded")

X=prepare_batch(md_file,demographic_data_norm)


test_predMD= modelMD.predict(X)
print ("MD model evaluation finished")

### RD model ######################################################
modelRD=model_from_json(model_json) 
weights_file = ensemblemodelpath+'\RD_MotorScore_weights.hdf5' # 
modelRD.load_weights(weights_file)

modelRD.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["categorical_accuracy"])
print ("RD model loaded")

X=prepare_batch(rd_file,demographic_data_norm)

test_predRD= modelRD.predict(X)
print ("RD model evaluation finished")
         
import numpy as np

ensemble_prediction = np.mean(
                [test_predWM, test_predFA],#test_predWM, test_predGM, test_predFA, test_predAD, test_predMD, test_predRD
                axis=0) 

test_pred2=np.argmax(ensemble_prediction,axis=1)
test_pred2 = np.array(test_pred2, 'float32')
  
if test_pred2==0:
    print("Upperlimb motor impairment is Poor...")
else:
     print("Upperlimb motor impairment is Good...")
            
print("PREDICTION FINISHED---------------------------------------------")
            
