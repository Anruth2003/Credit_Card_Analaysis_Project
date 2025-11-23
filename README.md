# 🏦 Bank GoodCredit — Credit Risk Modeling (PM-PR-0015)

This project presents a complete **credit risk prediction workflow** built using Python and Machine Learning.  
The notebook (`Bank_Goodcredit.ipynb`) includes data cleaning, preprocessing, feature engineering, model development, evaluation, and hyperparameter tuning.


---

## 📘 Project Contents

The notebook walks through the following major steps:

### **1. Install & Import Required Libraries**
Libraries used include:
- pandas, numpy  
- matplotlib, seaborn  
- scikit-learn  
- imbalanced-learn  
- XGBoost  
- TensorFlow / Keras  
- Optuna (hyperparameter tuning)

---

## 📂 Data Input

The project reads data from **CSV files**, including:
- Customer account data  
- Customer demographics  
- Customer enquiry data  

(Ensure your CSV files match the structure expected in the notebook.)

---

## 🔍 Exploratory Data Analysis (EDA)
Basic EDA steps include:
- Shape and structure of datasets  
- Null value checks  
- Summary statistics  
- Plotting distributions and relationships  

---

## 🧹 Data Cleaning & Preprocessing Pipeline

The notebook performs dataset-specific cleaning:

### **Customer Account Table**
- Handling missing values  
- Correcting inconsistent formats  
- Removing duplicates  

### **Customer Demographics Table**
- Fixing categorical values  
- Handling outliers  
- Missing value imputation  

### **Customer Enquiry Table**
- Dropping irrelevant columns  
- Aggregating recent enquiries  

---

## 🛠️ Feature Engineering

Includes:
- Creating new derived variables  
- Combining features across tables  
- Encoding categorical variables (OneHot / LabelEncoding)  
- Scaling numerical variables  
- Splitting features (X) and target (y)  
- Train-test split  

---

## 🤖 Machine Learning Models

### **Model 1 — XGBoost**
- Model training  
- Evaluation (accuracy, confusion matrix, classification report)  
- Feature importance ranking  
- Re-training using Top 20 features  

### **Model 2 — Neural Network (TensorFlow/Keras)**
- Label encoding  
- Oversampling (SMOTE)  
- Feature scaling  
- Model architecture definition  
- Model training  
- Evaluation metrics  

---

## 🎯 Hyperparameter Tuning (Optuna)

The notebook runs an Optuna optimization study to find optimal parameters for the neural network model:

- Best params extraction  
- Rebuilding the model using optimized parameters  
- Re-training and re-evaluating  
- Feature importance (Top 20 features)

---

## 📊 Evaluation Metrics

For both ML and NN models, the notebook reports:
- Accuracy  
- Confusion matrix  
- Classification report  
- Feature importance  
- Loss vs Accuracy plots (for neural network)  

---

## 🚀 How to Run

1. Install Python 3.8+  
2. Install required packages:
   ```bash
   pip install -r requirements.txt
3. Place your CSV files in the project directory
4. Open Jupyter Notebook:
   ```bash
   jupyter notebook Bank_Goodcredit.ipynb
5. Run all cells sequentially
