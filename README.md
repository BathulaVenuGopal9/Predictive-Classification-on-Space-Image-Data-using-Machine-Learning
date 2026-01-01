# Predictive-Classification-on-Space-Image-Data-using-Machine-Learning

## Image Preprocessing, Feature-Based Learning & Model Limitations Analysis
## Project Overview

This project implements a machine learning–based image classification system designed to classify space-related images (such as galaxies and other celestial image objects from hubble telescope) using classical supervised learning techniques.

The solution follows a complete pipeline involving image preprocessing, feature extraction, model training, evaluation, and inference, with a Decision Tree classifier selected as the best-performing baseline model.

The project also critically analyzes the limitations of traditional machine learning models on image data, laying a clear foundation for future enhancement using deep learning architectures.

## Problem Statement

Space image datasets contain complex visual patterns, textures, and spatial relationships. Accurately classifying such images requires capturing both local pixel-level features and global spatial structures.

The objective of this project is to:

Classify space images using classical machine learning techniques

Understand how feature-based models behave on image data

Identify limitations in prediction behavior

Propose technically sound improvements for future iterations

## Key Challenges in Space Image Classification
1. High Visual Complexity

Space images contain overlapping textures and noise

Subtle visual differences exist between object classes

Important information may exist only in small regions of the image

2. Image-to-Feature Conversion

Classical ML models cannot directly process raw images

Images must be converted into numerical feature vectors

Spatial relationships between pixels are often lost during flattening

3. Model Sensitivity to Local Pixel Patterns

Decision Tree models learn feature thresholds, not spatial structure

Presence of a small galaxy-like pixel region can dominate prediction

The model may classify an image as a galaxy even if the galaxy occupies only a minor portion of the image

## System Architecture
Raw Space Images
   ↓
Image Preprocessing
   ↓
Pixel-Based Feature Extraction
   ↓
Model Training
   ↓
Model Evaluation
   ↓
Model Serialization

## Image Preprocessing Pipeline

To prepare image data for machine learning, the following preprocessing steps were applied:

Image resizing to ensure consistent input dimensions

Conversion to grayscale (where applicable)

Pixel normalization for numerical stability

Flattening images into one-dimensional feature vectors

## Important Note:
Flattening images removes spatial context, meaning the model sees pixel intensity values without understanding object shape or position.

## Model Development
### Selected Model

Algorithm: Decision Tree Classifier

Learning Type: Supervised Machine Learning

Why Decision Tree?

Interpretable decision-making process

Ability to handle non-linear feature thresholds

Fast training and inference

Suitable as a baseline model for image-based classification

## Observed Model Behavior (Critical Insight)

During evaluation, the following behavior was observed:

If an image contains even a small portion of galaxy-like pixel patterns, the model frequently predicts the entire image as a galaxy.

Why This Happens (Technical Explanation)

Decision Trees evaluate individual feature thresholds

Pixel-based features representing galaxy textures trigger learned splits

The model does not understand spatial context or object boundaries

Local pixel similarity outweighs global image structure

This is a known limitation of classical ML models when applied to image data.

## Model Evaluation

The model was evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

While performance metrics indicate reasonable baseline accuracy, qualitative analysis revealed misclassification patterns related to spatial feature loss.

## Model Artifacts

dt_Space_data_project_best.pkl – trained Decision Tree model

label_encoder.pkl – encoded class labels

These artifacts enable reproducible inference and further experimentation.

## Technology Stack

Programming Language: Python

Image Processing: OpenCV, PIL

Machine Learning: Scikit-learn (Decision Tree)

Data Handling: NumPy

Model Persistence: Pickle

## Repository Structure
├── space_best_model_dt.ipynb
├── dt_Space_data_project_best.pkl
├── label_encoder.pkl
├── requirements.txt
├── README.md

## Key Outcomes

Built an end-to-end image classification pipeline using classical ML

Demonstrated how pixel-based features influence model predictions

Identified real-world limitations of Decision Trees on image data

Established a strong baseline for further improvement

## Future Enhancements (Planned)
### Transition to Deep Learning

To overcome the observed limitations, future versions of this project will adopt Deep Learning architectures, specifically:

Convolutional Neural Networks (CNNs) to capture spatial hierarchies

Automatic feature learning instead of manual pixel flattening

Improved robustness against partial object presence

Better generalization to complex celestial image structures

Additional Improvements

Data augmentation for better generalization

Class imbalance handling

Model explainability using activation maps

Deployment of a deep learning–based inference system

## Final Note

This project intentionally uses a classical machine learning approach to:

Establish a clear baseline

Analyze real limitations

Provide a technically justified transition path to deep learning

This reflects engineering maturity, not model weakness.

## Project Links
Hugging Face Live Demo (optional): **https://huggingface.co/spaces/venugopal99Bathula/Space-Object-Classification-with_Machine-Learning**

# Author

Bathula Venu Gopal
Data Science Intern @ Innomatics Research Labs
Former Amazon ML Data Associate
Focus Areas: Machine Learning, Computer Vision & Model Deployment
