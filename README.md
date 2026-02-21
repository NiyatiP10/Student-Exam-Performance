# Exam Score Prediction – Ridge + XGBoost Stacking Model

This project builds a high-performance regression pipeline to predict student **exam scores** using advanced feature engineering, target encoding, Ridge regression, and XGBoost with stacking.

---

## Project Overview

The goal of this project is to predict student exam scores (`exam_score`) using:

* Extensive feature engineering (34 engineered features)
* Target encoding for categorical variables
* Ridge Regression (as base model)
* XGBoost (as final model)
* 10-Fold Cross Validation
* Stacking (Ridge predictions used as meta-feature)

The final submission file is generated for evaluation.

---

## Dataset Files Used

The project expects the following files:

* `train.csv`
* `test.csv`
* `Exam_Score_Prediction.csv` (external/original dataset)
* `student_performance_dataset.csv`
* `sample_submission.csv`

---

## Feature Engineering

The function `preprocess_optimized()` creates **34 engineered numerical features**, including:

### Polynomial Features

* Squared terms for:

  * `study_hours`
  * `class_attendance`
  * `sleep_hours`
  * `age`

### Log & Sqrt Transformations

* `log_study_hours`
* `log_class_attendance`
* `log_sleep_hours`
* `sqrt_study_hours`
* `sqrt_class_attendance`

### Interaction Features

* Study × Attendance
* Study × Sleep
* Attendance × Sleep
* Age × Study

### Ratio Features

* `study_hours_over_sleep`
* `attendance_over_sleep`
* `attendance_over_study`

### Ordinal Encoding

* `sleep_quality`
* `facility_rating`
* `exam_difficulty`

### Rule-Based Flags

* High attendance & high study
* Ideal sleep flag
* High study flag

### Binned Features

* Study bins
* Attendance bins
* Sleep bins
* Age bins

---

## Modeling Strategy

### 1️. Ridge Regression (Base Model)

* Target encoding applied to categorical features
* 10-Fold KFold Cross Validation
* Alpha selection via `RidgeCV`
* Out-of-fold predictions saved
* Predictions clipped between 0–100

Ridge OOF RMSE is calculated.

---

### 2️. XGBoost (Final Model)

XGBoost is trained using:

* All base features
* All engineered features
* Ridge predictions as a **meta-feature**

### Key XGBoost Parameters

```python
n_estimators = 40000
learning_rate = 0.003
max_depth = 8
subsample = 0.82
colsample_bytree = 0.52
min_child_weight = 7
reg_lambda = 8.0
reg_alpha = 0.3
tree_method = "hist"
enable_categorical = True
early_stopping_rounds = 250
device = "cuda"
```

* 10-Fold CV
* Early stopping
* RMSE evaluation metric

---

## Outputs Generated

* `xgb_oof_optimized.csv` → Out-of-fold predictions
* `submission.csv` → Final test predictions
* Feature importance (Top 10 plotted using gain)

---

## Final Feature Summary

* Base Features: 11
* Engineered Features: 34
* Ridge Meta Feature: 1
* Total Features: 46

---

## Model Performance

The script prints:

* Ridge OOF RMSE
* XGBoost OOF RMSE
* Top 10 Most Important Features (Gain-based)

---

## Libraries Used

* numpy
* pandas
* matplotlib
* seaborn
* xgboost
* scikit-learn

---

## Key Highlights

1. Heavy feature engineering
2. Proper cross-validation
3. Target encoding
4. Model stacking
5. GPU training support
6. Early stopping for optimal trees

---

## How to Run

1. Place all required CSV files in the working directory.
2. Run the script in a Python environment (GPU recommended).
3. The submission file `submission.csv` will be generated.

---

## Author

Niyati Patil
(NiyatiP10)

