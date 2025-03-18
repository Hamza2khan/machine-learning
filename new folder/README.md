# THE UNIVERSITY OF AZAD JAMMU & KASHMIR, MUZAFFARABAD  

## 🏛 OPEN ENDED LAB - MACHINE LEARNING  
**DEPARTMENT OF SOFTWARE ENGINEERING**  

### 📜 Submission Details  
- **Submitted To:** Engr. Ahmed Khawaja  
- **Submitted By:** Hamza Shahzad,Hasham Ahmed,Shams Tabraiz 
- **Submitted On:** 30/02/2025  
- **Semester:** 5th  
- **Course Code:** SE-3105  
- **Roll No:** BS-2022-SE-09,02,36  

---

## 📖 Overview  
This project focuses on predicting Event-Free Survival (EFS) for patients post-Hematopoietic Cell Transplantation (HCT) by ensembling three advanced machine learning models: an Event-masked Pairwise Ranking Loss Neural Network (PRL-NN), a Yunbase model, and an EDA & Ensemble Model. The dataset is sourced from the Kaggle competition "Equity Post-HCT Survival Predictions." The project demonstrates data preprocessing, model training, out-of-fold (OOF) prediction generation, and an ensemble strategy to optimize performance, culminating in a final submission file evaluated on the competition leaderboard (LB):  

- **PRL-NN**: Achieved **LB score of 0.691** using a neural network with pairwise ranking loss and an XGBoost classifier mask.  
- **Yunbase**: Achieved **LB score of 0.689** with a custom ensemble of **LightGBM** and **CatBoost**.  
- **EDA & Ensemble**: Achieved **LB score of 0.689** with **Exploratory Data Analysis (EDA)** and a multi-target ensemble approach.  

The final **ensemble combines** these models' predictions using **rank-based weighting**, optimized via **cross-validation**.  

---

## 📂 Project Files  
- **PRL-NN**: Training and inference, outputs `submission2.csv`.  
- **Yunbase**: Uses `baseline.py`, outputs `submission1.csv`.  
- **EDA & Ensemble**: Multi-model training and analysis, outputs `submission3.csv`.  
- **Ensemble Notebook**: Combines predictions, outputs `submission.csv`.  

---

## 📊 Dataset  
- **Source**: Kaggle competition "Equity Post-HCT Survival Predictions".  
- **Size**: 28,800 training entries, 60 attributes.  
- **Test Set**: Matches submission requirements.  

### 🎯 Target Variables  
- `efs`: **Binary** (0 = event, 1 = survival).  
- `efs_time`: **Time-to-event** in months.  

### 🔑 Key Features  
- `prim_disease_hct`, `hla_match_b_low`, `prod_type`, `year_hct`, `obesity`, `donor_age`,  
  `prior_tumor`, `gvhd_proph`, `sex_match`, `comorbidity_score`, `karnofsky_score`,  
  `donor_related`, `age_at_hct`, `race_group`.  

---

## 🚀 Project Workflow  
The project is structured into four main phases:  

### **1️⃣ Individual Model Development**  
Training and inference for:  
- **PRL-NN**  
- **Yunbase**  
- **EDA & Ensemble models**  

### **2️⃣ Data Preprocessing**  
- **Feature Engineering**  
- **Handling Missing Values**  

### **3️⃣ Model Prediction**  
- **Generate Out-of-Fold (OOF) and Test Predictions**  

### **4️⃣ Ensemble Optimization**  
- **Combine predictions** using **rank-based weighting**  

---

## 🧑‍💻 Model Training Details  

### **🔹 PRL-NN (Event-masked Pairwise Ranking Loss Neural Network)**  

#### **📌 Data Loading and Preprocessing**  
```python
train = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/train.csv")
test = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/test.csv")

train = preprocess_data(train)  # Fill NA, encode categoricals
test = preprocess_data(test)

train = features_engineering(train)  # Add 'donor_age_diff', 'hla_mismatch_sum'
test = features_engineering(test)
```

#### **📌 XGBoost Classifier**  
```python
model_xgb = XGBClassifier(max_depth=4, n_estimators=10_000, learning_rate=0.03, device="cuda")
model_xgb.fit(x_train, y_train, eval_set=[(x_valid, y_valid)], verbose=100)

oof_xgb = model_xgb.predict_proba(x_valid)[:, 1]
pred_efs = model_xgb.predict_proba(x_test)[:, 1]
```

---

### **🔹 Yunbase Model**  

#### **📌 Data Preprocessing**  
```python
train['donor_age_diff'] = train['donor_age'] - train['age_at_hct']
train['target'] = transform_survival_probability(train, 'efs_time', 'efs')

train = FE(train)  # Feature engineering: 'nan_value_each_row', cross features
```

#### **📌 Model Training**  
```python
yunbase = Yunbase(num_folds=5, models=[(LGBMRegressor(), 'lgb'), (CatBoostRegressor(), 'cat')], FE=FE)
yunbase.fit(train, category_cols=nunique2)
```

---

### **🔹 Ensemble Strategy**  

#### **📌 Ranking and Weighting**  
```python
rank1 = rankdata(sub1['prediction'])  # Yunbase
rank2 = rankdata(sub2['prediction'])  # PRL-NN
rank3 = rankdata(sub3['prediction'])  # EDA

for w1 in [0.30, 0.32, 0.34]:
    for w2 in [0.33, 0.35, 0.37]:
        w3 = 1 - w1 - w2
        y_pred['prediction'] = w1 * rank1 + w2 * rank2 + w3 * rank3
        temp_score = score(y_true, y_pred, 'ID')
```
---

## 📈 Results  
- **PRL-NN**: LB Score **0.691**  
- **Yunbase**: LB Score **0.689**  
- **EDA & Ensemble**: LB Score **0.689**  
- **Final Ensemble**: Optimized CV score **~0.69+**, aiming for leaderboard improvement  

### **📊 Stratified C-Index Scores**  
- **Cox**: `0.6568`  
- **Kaplan-Meier**: `0.9983`  
- **Nelson-Aalen**: `0.9983`  

---

## 🔧 Improvements  
- **Hyperparameter Tuning**  
  - Optimize PRL-NN epochs, XGBoost depth, or Yunbase parameters.  
- **Feature Engineering**  
  - Add interaction terms like `comorbidity_score * donor_age`.  
- **Ensemble Strategy**  
  - Explore stacking or blending instead of rank-based weighting.  

---

## 🛠️ How to Run  
### **Environment Setup**  
- **Platform**: Kaggle Notebook  
- **Python Version**: 3.10.12  
- **GPU**: Enabled  

### **Required Libraries**  
`pandas`, `numpy`, `torch`, `xgboost`, `lightgbm`, `catboost`, `lifelines`, `pytorch_lightning`, `pytorch_tabular`, `sklearn`, `plotly`.  

### **Execution Steps**  
1. Run **PRL-NN** sections to generate `submission2.csv`.  
2. Run **Yunbase** sections (ensure `baseline.py` is copied) to generate `submission1.csv`.  
3. Run **EDA & Ensemble** sections to generate `submission3.csv`.  
4. Run **Ensemble Notebook** section to combine results into `submission.csv`.  
