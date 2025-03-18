# THE UNIVERSITY OF AZAD JAMMU & KASHMIR, MUZAFFARABAD  

## 🏛 OPEN ENDED LAB - MACHINE LEARNING  
**DEPARTMENT OF SOFTWARE ENGINEERING**  

### 📜 Submission Details  
- **Submitted To:** Engr. Awais Rathore  
- **Submitted By:** Hamza Shahzad  
- **Submitted On:** 30/02/2025  
- **Semester:** 5th  
- **Course Code:** SE-3105  
- **Roll No:** BS-2022-SE-09  

---

# 📖 Open Ended Lab Report

## 1️⃣ Introduction  
The **MNIST dataset** consists of grayscale images of handwritten digits **(0-9)** in a **28x28 pixel format**.  
This project aims to classify these digits using different machine learning models and a deep learning model.  
The dataset has been preprocessed into **CSV files (`mnist_train.csv` and `mnist_test.csv`)**, with each image represented as a **1D vector of 784 features**.

---

## 2️⃣ Methodology  

### **2.1 Data Preprocessing**  
- The dataset is **loaded** from CSV files using `pandas`.  
- Features are **scaled** using `StandardScaler` and `MinMaxScaler`.  
- **Principal Component Analysis (PCA)** is applied to reduce dimensionality.  
- Missing values are handled using `SimpleImputer`.  
- The dataset is **split** into training and testing sets using `train_test_split`.  

### **2.2 Models Used**  
The following models were implemented and evaluated:  
✅ **Logistic Regression**  
✅ **k-Nearest Neighbors (KNN)**  
✅ **Decision Tree**  
✅ **Naïve Bayes**  
✅ **Artificial Neural Network (ANN)**  

---

## 3️⃣ Results  

### **🔹 Logistic Regression Accuracy: 85.4%**  
```plaintext
Precision: 0.86, Recall: 0.85, F1-Score: 0.85  
Confusion Matrix Available in Report  
```

### **🔹 KNN Accuracy: 94.8%**  
```plaintext
Precision: 0.95, Recall: 0.95, F1-Score: 0.95  
Confusion Matrix Available in Report  
```

### **🔹 Decision Tree Accuracy: 83.6%**  
```plaintext
Precision: 0.84, Recall: 0.83, F1-Score: 0.83  
Confusion Matrix Available in Report  
```

### **🔹 Naïve Bayes Accuracy: 45.9%**  
```plaintext
Precision: 0.56, Recall: 0.45, F1-Score: 0.41  
Confusion Matrix Available in Report  
```

### **🔹 ANN Accuracy: 97.3%**  
```plaintext
Precision: 0.97, Recall: 0.97, F1-Score: 0.97  
Confusion Matrix Available in Report  
```

### **🔹 Tuned ANN Accuracy: 97.2%**  
```plaintext
Best Parameters: {'hidden_layer_sizes': (128,), 'learning_rate_init': 0.001}  
Precision: 0.97, Recall: 0.97, F1-Score: 0.97  
Confusion Matrix Available in Report  
```

---

## 4️⃣ Discussion  
The **best-performing model** was the **Tuned ANN**, achieving an accuracy of **97%**.  
The **neural network** performed well due to its ability to learn complex patterns in the dataset.  
Traditional models like **Decision Trees** and **Naïve Bayes** had lower performance due to their limitations in handling high-dimensional data.  

---

## 5️⃣ Conclusion  
This project successfully implemented multiple machine learning models for classifying **MNIST digits**.  
The best-performing model was the **Tuned ANN with an accuracy of 97%**.  
Future work could involve using **Convolutional Neural Networks (CNNs)** for even better image recognition performance.  

---
