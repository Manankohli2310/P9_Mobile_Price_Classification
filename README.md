# 📱 Mobile Price Range Classification using Machine Learning

## 🚀 Project Overview

This project predicts the **price range of a smartphone** based on its technical specifications using Machine Learning algorithms. The application classifies a mobile phone into one of four categories:

* Low Cost
* Medium Cost
* High Cost
* Very High Cost

The project compares **Decision Tree** and **Random Forest** classifiers and selects the best-performing model for deployment through an interactive Streamlit web application.

---

## 🌐 Live Demo

### Streamlit Application

[Add Your Streamlit Link Here]

### YouTube Project Demo

[Add Your YouTube Demo Link Here]

---

## 🎯 Problem Statement

Smartphones come with a wide variety of specifications such as RAM, battery capacity, processor speed, storage, camera quality, and display resolution. Determining the expected price category based on these specifications can be useful for manufacturers, retailers, and customers.

This project uses Machine Learning to automatically predict the likely price range of a mobile device using its hardware specifications.

---

## 📊 Dataset Information

Dataset: Mobile Price Classification Dataset

### Dataset Statistics

* Total Records: 2000
* Features Used: 14
* Target Classes: 4
* Missing Values: 0

### Target Classes

| Class | Price Range    |
| ----- | -------------- |
| 0     | Low Cost       |
| 1     | Medium Cost    |
| 2     | High Cost      |
| 3     | Very High Cost |

---

## 🔍 Features Used

The final model was trained using the following features:

* Battery Power
* Clock Speed
* Front Camera
* Internal Memory
* Mobile Depth
* Mobile Weight
* Number of CPU Cores
* Primary Camera
* Screen Resolution Height
* Screen Resolution Width
* RAM
* Screen Height
* Screen Width
* Talk Time

### Features Removed

The following features were removed after feature importance analysis because they contributed very little to prediction performance:

* Bluetooth Support
* Dual SIM Support
* 3G Support
* 4G Support
* Touch Screen Support
* WiFi Support

---

## 🤖 Machine Learning Algorithms Used

### 1. Decision Tree Classifier

A Decision Tree creates a series of decision rules based on feature values and classifies the mobile phone into a price category.

**Advantages**

* Easy to understand
* Easy to visualize
* Fast prediction

**Accuracy Achieved**

* 84.75%

---

### 2. Random Forest Classifier

Random Forest combines multiple Decision Trees and makes predictions using majority voting.

**Advantages**

* Better generalization
* Reduced overfitting
* Higher accuracy
* More robust predictions

**Accuracy Achieved**

* 90.50%

---

## 🏆 Best Model

Random Forest was selected as the final model because it achieved the highest accuracy.

| Model         | Accuracy |
| ------------- | -------- |
| Decision Tree | 84.75%   |
| Random Forest | 90.50%   |

---

## 📈 Feature Importance Analysis

Feature importance analysis revealed that the most influential features for mobile price prediction are:

1. RAM
2. Battery Power
3. Screen Resolution Width
4. Screen Resolution Height
5. Internal Memory

RAM was found to be the strongest predictor of smartphone price range.

---

## 🖥️ Streamlit Application Features

The Streamlit web application includes:

### Home Page

* Project overview
* Model performance metrics

### Dataset Insights

* Dataset statistics
* Feature information
* Target class explanation

### Decision Tree Results

* Accuracy
* Classification report
* Confusion matrix

### Random Forest Results

* Accuracy
* Classification report
* Confusion matrix

### Feature Importance

* Visual feature importance chart
* Model interpretation

### Mobile Price Prediction

* User-friendly specification input
* Price category prediction
* Prediction confidence scores

---

## 📂 Project Structure

```text
Mobile_Price_Classification/
│
├── data/
│   └── mobile_price.csv
│
├── models/
│   ├── decision_tree_model.pkl
│   ├── random_forest_model.pkl
│   └── selected_features.pkl
│
├── reports/
│   ├── classification_report_dt.csv
│   ├── classification_report_rf.csv
│   ├── confusion_matrix_dt.csv
│   ├── confusion_matrix_rf.csv
│   ├── feature_importance.csv
│   └── model_comparison.csv
│
├── images/
│   ├── confusion_matrix_dt.png
│   ├── confusion_matrix_rf.png
│   └── feature_importance.png
│
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Joblib
* Streamlit

---

## 📋 Future Improvements

* Hyperparameter tuning using GridSearchCV
* Cross-validation based model evaluation
* Support for modern smartphone specifications
* Additional ensemble algorithms comparison
* Mobile-friendly UI enhancements
* Deployment using cloud services

---

## ⚠️ Limitations

* Dataset contains only 2000 samples.
* Some features are dataset-specific and do not directly correspond to modern smartphone specifications.
* The model predicts price categories rather than exact prices.
* Predictions may become less reliable for specifications far outside the dataset range.
* Market trends and brand value are not considered.

---