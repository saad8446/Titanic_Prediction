# Titanic: Machine Learning from Disaster - Preprocessing Pipeline

This repository contains the data preprocessing, exploratory data analysis (EDA), and feature engineering pipeline for the classic Kaggle Titanic dataset. The goal of this phase was to clean raw, messy passenger data and transform it into a pristine numerical matrix optimized for Machine Learning algorithms.

## 📌 Project Overview
Before training any machine learning models, data must be cleaned, structured, and scaled. This project takes the raw Titanic passenger manifest and handles missing values, encodes categorical text into numerical data, engineers new predictive features, and pre-scales data for distance-based algorithms.

---

## 📊 Phase 1: Exploratory Data Analysis (EDA)
To understand the dataset before manipulating it, the following visualizations were implemented to uncover underlying patterns:
* **Target Distribution Plot:** Analyzed the baseline survival vs. non-survival rates.
* **Survival Rate by Gender:** Visualized the strong correlation between gender (`Sex`) and survival chance ("women and children first").
* **Age Density Plot (KDE):** Highlighted the higher survival density among infants and young children.
* **Correlation Heatmap:** Mapped linear relationships between all numerical features to prevent feature redundancy.

---

## 🛠️ Phase 2: Data Preprocessing & Cleaning

### 1. Handling Missing Values
* **`Age`:** Imputed missing values using the column **mean** to preserve the dataset's statistical distribution. 
* **`Embarked`:** Dropped rows with missing values since they represented less than 0.5% of the data.
* **`Cabin`:** Extracted value by converting it into a binary feature (`Has_Cabin`), representing whether a passenger had a designated cabin or not.

### 2. Feature Engineering
New features were created to extract deeper insights for the predictive models:
* **`Family_Size`:** Combined `SibSp` (siblings/spouses) and `Parch` (parents/children) variables ($Family\_Size = SibSp + Parch + 1$).
* **`Is_Alone`:** A binary flag ($0$ or $1$) derived from `Family_Size` indicating if a passenger traveled completely alone.

### 3. Categorical Encoding (One-Hot Encoding)
* Converted text features (`Sex`, `Embarked`) into numerical format using `pd.get_dummies()`.
* Applied `drop_first=True` to avoid the **Dummy Variable Trap** (multicollinearity). For example, dropping `Embarked_C` makes Cherbourg the mathematical reference baseline when both `Embarked_Q` and `Embarked_S` are `0`.

### 4. Feature Scaling (Standardization)
* Continuous columns with drastically different ranges (`Age` and `Fare`) were scaled using `StandardScaler`. 
* This shifts their distributions to have a mean of 0 and a standard deviation of 1 ($\mu=0, \sigma=1$), ensuring distance-based algorithms (like Logistic Regression or KNN) treat them fairly.

### 5. Column Dropping
Removed redundant or non-predictive features that would cause overfitting: `PassengerId`, `Name`, `Ticket`, `SibSp`, and `Parch`.

---

## 📂 Final Dataset Structure
The preprocessed dataset is saved as `titanic_cleaned_for_ml.csv` and contains purely numerical values ready for ingestion.

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `Survived` | int (0 or 1) | **Target Variable** (0 = Died, 1 = Survived) |
| `Pclass` | int (1, 2, or 3) | Ticket class |
| `Age` | float (Scaled) | Standardized passenger age |
| `Fare` | float (Scaled) | Standardized ticket fare |
| `Has_Cabin` | int (0 or 1) | 1 if passenger had a cabin, 0 if NaN |
| `Family_Size`| int | Total number of family members on board |
| `Is_Alone` | int (0 or 1) | 1 if traveling alone, 0 if with family |
| `Sex_male` | int (0 or 1) | 1 = Male, 0 = Female |
| `Embarked_Q` | int (0 or 1) | 1 = Boarded at Queenstown |
| `Embarked_S` | int (0 or 1) | 1 = Boarded at Southampton |

*Note: If both `Embarked_Q` and `Embarked_S` are 0, the passenger boarded at Cherbourg (`C`).*

---

## 🚀 Model Training 
1. Splitting the cleaned data into Training and Testing sets (`train_test_split`).
2. Training and comparing multiple Machine Learning models:
   * **Distance-based:** Logistic Regression, K-Nearest Neighbors (KNN)
3. Evaluating performance using Accuracy, Precision, Recall, and F1-Score.

