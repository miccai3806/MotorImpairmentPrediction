# Attention-Based Multimodal Deep Learning Model for Post-Stroke Motor Impairment Prediction

This repository contains the official implementation of the MICCAI-2025 study:
"Attention-Based Multimodal Deep Learning Model for Post-Stroke Motor Impairment Prediction".

# Overview
This study introduces an attention-based multimodal deep learning model to predict post-stroke motor impairment using Diffusion Tensor Imaging (DTI) and structural MRI features. The model integrates fractional anisotropy (FA), mean diffusivity (MD), radial diffusivity (RD), axial diffusivity (AD), white matter (WM), and gray matter (GM) intensity maps through a 3D convolutional neural network (3D-CNN) with residual connections and the Convolutional Block Attention Module (CBAM).
An ensemble learning approach combined multiple modalities, achieving high classification accuracy for motor impairment prediction.
 
# Repository Structure
📂 Ensemble_Models/ – Contains pre-trained model weights for multimodal prediction.
📂 Test_Data/ – Includes test data from two individuals, with available AD, FD, MD, RD, WM, and GM maps.
📂 utils/ – Python scripts for data preprocessing, network creation, and metric computation.
📄 motorscores_train.py – Training script for the 3D-CNN models.
📄 motorscore_ensemble.py – Script for ensemble model integration across different modalities.
📄 motorscore_ensemble_prediction.py – Prediction script for evaluating new test samples.
