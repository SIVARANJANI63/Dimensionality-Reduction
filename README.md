# Human Activity Recognition using Smartphones (UCI HAR Dataset)

This project focuses on Human Activity Recognition (HAR) using smartphone sensor data. The dataset contains readings from accelerometers and gyroscope sensors, and the goal is to classify various activities based on these sensor readings.

## Table of Contents
- [Overview](#overview)
- [Requirements](#requirements)
- [Dataset](#dataset)
- [Installation](#installation)
- [Steps and Workflow](#steps-and-workflow)
  - [1. Load Data](#1-load-data)
  - [2. Exploratory Data Analysis (EDA)](#2-exploratory-data-analysis-eda)
  - [3. Label Encoding](#3-label-encoding)
  - [4. Data Preprocessing](#4-data-preprocessing)
  - [5. Model Training and Evaluation](#5-model-training-and-evaluation)
  - [6. Dimensionality Reduction using K-Means](#6-dimensionality-reduction-using-k-means)
  - [7. Final Model Evaluation](#7-final-model-evaluation)
- [Future Enhancements](#future-enhancements)
- [Acknowledgements](#acknowledgements)

## Overview
This repository contains code for building and evaluating models for classifying human activities using smartphone sensor data. The dataset includes 561 features derived from sensor readings, with a corresponding class label for each observation representing one of six activities.

The project demonstrates the following steps:
1. **Data Preprocessing**: Downloading and cleaning the dataset.
2. **Exploratory Data Analysis (EDA)**: Understanding the data and checking for inconsistencies.
3. **Feature Selection**: Reducing dimensionality using K-Means clustering.
4. **Model Training**: Training machine learning models (Naive Bayes, K-Means) and evaluating their performance.

## Requirements
The following libraries are required for running the code:

- `requests`
- `beautifulsoup4`
- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `time`

Install the dependencies using `pip`:

```bash
pip install requests beautifulsoup4 pandas numpy scikit-learn matplotlib seaborn

```

## Dataset

The dataset used in this project is the **Human Activity Recognition using Smartphones** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

### Dataset Details:
- **561 Features**: Derived from accelerometer and gyroscope data.
- **6 Classes**: Representing activities:
  - Walking
  - Walking Upstairs
  - Walking Downstairs
  - Sitting
  - Standing
  - Laying

The dataset is provided as a `.zip` file containing two main sets of data:
- `train`: Contains training samples.
- `test`: Contains test samples.

### Dataset Files:
- `X_train.txt`: Features for training data.
- `y_train.txt`: Labels for training data.
- `X_test.txt`: Features for testing data.
- `y_test.txt`: Labels for testing data.

## Installation

Clone this repository:

```bash
git clone https://github.com/SIVARANJANI63/Dimensionality-Reduction.git
cd har-smartphone-recognition
```

## Steps and Workflow

### 1. Load Data
The first step is to download and load the dataset using the `requests` library to fetch the dataset from the UCI repository. The `pandas` library is used to load the data into DataFrames.

### 2. Exploratory Data Analysis (EDA)
Perform basic EDA to understand the structure of the dataset, check for missing values, and examine the statistical summary of the features.

**EDA steps:**
- Summary statistics of the features.
- Check for missing values.
- Check the data types of the columns.

### 3. Label Encoding
The `LabelEncoder` from `sklearn` is used to encode the class labels into numeric values for machine learning algorithms.

```python
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
encoded_y = label_encoder.fit_transform(y.values.ravel())
```
## 4. Data Preprocessing
The dataset is scaled using `StandardScaler` to normalize the features to a standard scale. This helps improve the performance and convergence of machine learning algorithms.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
```
## 5. Model Training and Evaluation
We use a Naive Bayes model to classify the activities. The model is trained using the training data and evaluated on the test data. The accuracy of the model is calculated using the `accuracy_score` from `sklearn`.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

model = GaussianNB()
model.fit(X_train_full, y_train)
y_pred = model.predict(X_test_full)
accuracy = accuracy_score(y_test, y_pred)
```

## 6. Dimensionality Reduction using K-Means
We use K-Means clustering to reduce the dimensionality of the data by selecting representative features. This helps in improving the model's performance and reducing computation time.

```python
from sklearn.cluster import KMeans
import numpy as np

kmeans = KMeans(n_clusters=50, random_state=42)
kmeans.fit(df_scaled.T)  # Treat features as data points
selected_features_indices = np.argmin(kmeans.transform(df_scaled.T), axis=0)
selected_features = df_scaled[:, selected_features_indices]
```

## 7. Final Model Evaluation
After reducing the dimensionality, we train the model again with the selected features and evaluate the accuracy and runtime.

```python
model1 = GaussianNB()
model1.fit(X_train_full, y_train)
y_pred = model1.predict(X_test_full)
```

## Future Enhancements

- **Model Comparison**: Integrate other models like Random Forest and Support Vector Machine (SVM) for comparison to evaluate the performance difference.
  
- **Improved Feature Engineering**: Explore additional techniques for feature extraction from raw sensor data, such as temporal feature extraction or advanced signal processing.
  
- **Web Application**: Deploy the model as a web application, allowing for interactive inputs and real-time activity recognition.

## Acknowledgements

- **UCI Machine Learning Repository**: For providing the dataset.
  
- **Scikit-learn**: For the machine learning algorithms and preprocessing tools.
  
- **Matplotlib & Seaborn**: For data visualization.
  
- **KMeans**: For dimensionality reduction via clustering.
