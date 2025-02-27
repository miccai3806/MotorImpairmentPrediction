# Attention-Based Multimodal Deep Learning Model for Post-Stroke Motor Impairment Prediction

This repository contains the official implementation of the MICCAI-2025 study:
"Attention-Based Multimodal Deep Learning Model for Post-Stroke Motor Impairment Prediction".

# Overview
This study introduces an attention-based multimodal deep learning model to predict post-stroke motor impairment using Diffusion Tensor Imaging (DTI) and structural MRI features. The model integrates fractional anisotropy (FA), mean diffusivity (MD), radial diffusivity (RD), axial diffusivity (AD), white matter (WM), and gray matter (GM) intensity maps through a 3D convolutional neural network (3D-CNN) with residual connections and the Convolutional Block Attention Module (CBAM).
An ensemble learning approach combined multiple modalities, achieving high classification accuracy for motor impairment prediction.
 
# Repository Structure
📂 Ensemble_Models/ – Contains pre-trained model weights for multimodal prediction.<br>
📂 Test_Data/ – Includes test data from two individuals, with available AD, FD, MD, RD, WM, and GM maps.<br>
📂 utils/ – Python scripts for data preprocessing, network creation, and metric computation.<br>
📄 motorscores_train.py – Training script for the 3D-CNN models.<br>
📄 motorscore_ensemble.py – Script for ensemble model integration across different modalities.<br>
📄 motorscore_ensemble_prediction.py – Prediction script for evaluating new test samples.<br>
# Getting Started
<b> Installation <br>
1. Clone the repository: <br>
   `git clone https://github.com/miccai3806/MotorImpairmentPrediction.git`<br>
   `cd strokeanalysis`<br>
2. Install dependencies<br>
   `pip install -r requirements.txt`<br>
3. Download and extract the pre-trained models inside the Ensemble_Models/folder.<br>

# Usage
<b> 1. Training the Model<br>
To train the 3D-CNN model, run:<br>
