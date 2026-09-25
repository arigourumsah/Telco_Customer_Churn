# Telco Customer Churn Prediction

This project builds a binary churn classifier to predict whether a telecom customer will leave the service based on their account and usage data. It benchmarks several standard classification algorithms, Logistic Regression, SVM, Decision Tree, Random Forest, KNN, Naive Bayes, LDA, and XGBoost, on a four-feature subset selected via chi-square ranking, then applies SMOTE oversampling and hyperparameter tuning via `GridSearchCV` to improve churn detection.

---

## Project Overview

The goal of this project is to classify telecom customers as churned or retained using structured account and service data. The workflow covers exploratory data analysis, feature engineering, encoding, chi-square feature selection, baseline model benchmarking, SMOTE oversampling, hyperparameter tuning, and final evaluation.

Key aspects:

- Binary churn classification: Churned (1) vs. Retained (0).
- End-to-end pipeline from raw data to tuned model evaluation.
- Feature selection with `SelectKBest(chi2, k=4)` to expose the most informative predictors.
- Three-stage evaluation: cross-validated baseline → SMOTE-resampled baseline → hyperparameter-tuned test set evaluation with ROC AUC, accuracy, and F1-score.

---

## Business Value

By accurately identifying customers likely to churn before they leave, this project helps telecom providers take proactive retention actions, such as targeted offers or support interventions, rather than reacting after the fact. Reducing churn directly improves revenue, lowers customer acquisition costs, and informs longer-term contract and pricing strategies. The interpretable four-feature model also makes it practical to deploy and explain predictions to business stakeholders.

---

## Dataset

The dataset is stored in `data/raw/customer_churn_data.csv` and contains **7,043** telecom customer records across 21 columns.

- Number of records: 7,043 (5,174 Retained, 1,869 Churned).
- Target: `Churn` (`Yes` / `No`, mapped to `1` / `0` for modeling).

The raw columns are:

| Column | Type | Description |
|---|---|---|
| `customerID` | object | Unique customer identifier (dropped before modeling) |
| `gender` | object | Customer gender |
| `SeniorCitizen` | int | Whether the customer is a senior (1 = Yes, 0 = No) |
| `Partner` | object | Whether the customer has a partner |
| `Dependents` | object | Whether the customer has dependents |
| `tenure` | int | Number of months the customer has been with the company |
| `PhoneService` | object | Whether the customer has phone service |
| `MultipleLines` | object | Whether the customer has multiple lines |
| `InternetService` | object | Internet service type (DSL, Fiber optic, No) |
| `OnlineSecurity` | object | Whether the customer has online security add-on |
| `OnlineBackup` | object | Whether the customer has online backup add-on |
| `DeviceProtection` | object | Whether the customer has device protection add-on |
| `TechSupport` | object | Whether the customer has tech support add-on |
| `StreamingTV` | object | Whether the customer has streaming TV |
| `StreamingMovies` | object | Whether the customer has streaming movies |
| `Contract` | object | Contract term (Month-to-month, One year, Two year) |
| `PaperlessBilling` | object | Whether the customer uses paperless billing |
| `PaymentMethod` | object | Payment method used |
| `MonthlyCharges` | float | Monthly charge amount |
| `TotalCharges` | object | Total charges (loaded as object; contains blanks for tenure = 0 rows) |
| `Churn` | object | Whether the customer churned |

> Note: `TotalCharges` is loaded as an object column due to blank entries for customers with `tenure = 0`. This is handled in the preprocessing step.

---

## Repository Structure

```text
Telco_Customer_Churn/
├─ data/
│  └─ raw/
│     └─ customer_churn_data.csv   # Original labeled customer dataset.
├─ images/
├─ notebooks/
│  └─ telco_customer_churn_prediction.ipynb   # Full analysis, feature engineering, and modeling pipeline.
├─ requirements.txt                            # Python dependencies.
└─ README.md                                   # Project documentation (this file).
```

---

## Methodology

### 1. Data Understanding

I begin by loading the dataset and inspecting its structure using `info()`, `head()`, and `describe()`. This confirms 7,043 rows and 21 columns. The numerical summary shows an average `tenure` of 32.37 months and an average `MonthlyCharges` of 64.76.

At this stage, I also observe that `TotalCharges` is stored as an object rather than a numeric column, and that several rows with `tenure = 0` contain blank `TotalCharges` values, a data quality issue I address before modeling.

---

### 2. Exploratory Data Analysis

EDA is performed in `notebooks/telco_customer_churn_prediction.ipynb` to understand the distribution of features and how they relate to churn.

#### 2.1 Data Distribution
![Data Distribution](/images/data_distribution.png)

I inspect the spread of the four main numerical features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`. `SeniorCitizen` is highly imbalanced since most customers are non-seniors. `tenure` is bimodally distributed with peaks at the beginning and end of the service lifetime. `MonthlyCharges` is broadly spread across service plan levels, and `TotalCharges` is right-skewed because many customers have low accumulated billing while longer-tenure customers accumulate much higher totals.

#### 2.2 Boxplot Analysis of Numerical Features
![Boxplots](/images/boxplots.png)

I compare `tenure`, `MonthlyCharges`, and `TotalCharges` against the churn label. `tenure` shows the strongest separation: churned customers have a much lower median tenure concentrated near the beginning of the customer lifecycle, while retained customers have a wider and higher tenure distribution. `MonthlyCharges` shows that churned customers tend to have higher monthly costs, and `TotalCharges` reflects the tenure effect, retained customers accumulate larger totals even though churned customers also show high-value outliers.

#### 2.3 Correlation Heatmap
![Correlation Heatmap](/images/correlation_heatmap.png)

The heatmap focuses on `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`. I observe a strong positive correlation of **0.83** between `tenure` and `TotalCharges`, a moderate positive correlation of **0.65** between `MonthlyCharges` and `TotalCharges`, and a negative correlation of **-0.35** between `tenure` and `Churn`. `MonthlyCharges` shows a weak positive relationship with `Churn` (0.19) and `TotalCharges` a weak negative one (-0.20), confirming that tenure is the strongest linear signal in the numerical set.

#### 2.4 Countplots of Categorical Features
![Countplots](/images/countplots.png)

I visualize the raw frequency distribution of all categorical variables. The dataset is roughly balanced by `gender`. Most customers do not have `Dependents`. Internet service is dominated by `Fiber optic` and `DSL`. `Month-to-month` is the most common contract type. `PaperlessBilling` leans heavily toward `Yes`, and `Electronic check` is the most common payment method.

#### 2.5 Proportion of Churn Rate in Categorical Features
![Churn Proportions](/images/proportions.png)

I plot stacked proportion charts to compare churn share across each category. The clearest pattern is in `Contract`: `Month-to-month` customers churn at approximately **43%**, `One year` at **11%**, and `Two year` at only **3%**. I also find elevated churn among customers without `OnlineSecurity`, `TechSupport`, or `DeviceProtection`, among `Fiber optic` internet users, and among those using `Electronic check` as their payment method.

---

### 3. Feature Engineering

After EDA, I prepare the features for modeling:

- Map `Churn` from `Yes` / `No` to `1` / `0`.
- Drop `customerID` from the feature table.
- Remap `SeniorCitizen` from `1` / `0` to `Yes` / `No` for consistent categorical handling.

---

### 4. One Hot Encoding and Label Encoding

I apply one-hot encoding to the following categorical variables: `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`, and `PaymentMethod`.

For `Contract`, I use manual ordinal encoding to preserve the natural ordering of commitment length:

| Contract | Encoded Value |
|---|---|
| Month-to-month | 0 |
| One year | 1 |
| Two year | 2 |

---

### 5. Feature Selection

I run chi-square feature selection using `SelectKBest(chi2, k=4)` on the full engineered feature matrix. The four selected predictors are:

- `tenure`
- `Contract`
- `MonthlyCharges`
- `TotalCharges`

This gives a compact, high-signal feature set that is easier to interpret and compare across models.

---

### 6. Train–Test Split

I split the selected feature matrix using `train_test_split` with `test_size = 0.2`, `random_state = 42`, and `stratify = y` to maintain the class distribution between training and test sets.

---

### 7. Scaling

I standardize the four selected features using `StandardScaler`. The scaler is fit on the training set and applied to both training and test sets to prevent data leakage.

---

### 8. Machine Learning Modeling

The modeling phase consists of three stages: a cross-validated baseline benchmark on the original imbalanced data, a second benchmark after SMOTE oversampling, and a final hyperparameter-tuned evaluation on the held-out test set.

#### 8.1 Baseline — Cross-Validated Benchmark (Before SMOTE)

I run stratified cross-validation on six classification algorithms: Logistic Regression, Random Forest, Decision Tree, SVM, Naive Bayes, and KNN. Each model is evaluated using mean ROC AUC and mean accuracy across folds.

| Algorithm | ROC AUC Mean | ROC AUC STD | Accuracy Mean | Accuracy STD |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.73 | 1.97 | 72.65 | 2.20 |
| SVM | 81.04 | 2.12 | 70.57 | 2.46 |
| Naive Bayes | 80.83 | 1.53 | 70.43 | 2.36 |
| Random Forest | 79.37 | 2.29 | 76.50 | 1.77 |
| KNN | 78.19 | 1.86 | 77.23 | 1.55 |
| Decision Tree | 65.48 | 2.73 | 72.38 | 2.08 |

Logistic Regression achieves the highest cross-validated ROC AUC (82.73) on the raw imbalanced data. Decision Tree is the weakest performer on ROC AUC.

#### 8.2 SMOTE Oversampling

Because the original training data is imbalanced (4,139 retained vs. 1,495 churned after the split), I apply SMOTE to oversample the minority churn class. After SMOTE, both classes are balanced at 4,139 samples each in the training set. I then re-run the same cross-validated benchmark on the resampled training data.

| Algorithm | Accuracy Mean | Accuracy STD |
|---|---:|---:|
| Random Forest | 80.77 | 4.44 |
| KNN | 77.70 | 4.67 |
| Decision Tree | 77.23 | 3.34 |
| Logistic Regression | 74.92 | 4.05 |
| SVM | 74.67 | 6.20 |
| Naive Bayes | 72.05 | 6.95 |

After SMOTE, Random Forest improves noticeably and leads on accuracy mean (80.77). Logistic Regression drops relative to its pre-SMOTE position, suggesting that oversampling benefits tree-based methods more on this feature set.

#### 8.3 Hyperparameter Tuning with GridSearchCV

I apply `GridSearchCV` to tune the top-performing models from the SMOTE benchmark on the held-out test set. The final evaluation reports ROC AUC, accuracy, and F1-score for each tuned model.

| Model | ROC AUC | Accuracy | F1-Score |
|---|---:|---:|---:|
| SVC | 0.75 | 0.72 | 0.61 |
| Logistic Regression | 0.75 | 0.72 | 0.60 |
| Gaussian Naive Bayes | 0.74 | 0.68 | 0.59 |
| Random Forest | 0.70 | 0.75 | 0.56 |
| KNN | 0.69 | 0.72 | 0.54 |
| Decision Tree | 0.69 | 0.71 | 0.54 |

After tuning, SVC and Logistic Regression tie for the highest ROC AUC (0.75) and accuracy (0.72). Random Forest achieves the best accuracy (0.75) among all tuned models but at a lower ROC AUC (0.70). F1-score is highest for SVC (0.61), which reflects a better balance between precision and recall on the churn class.

---

## Results Summary

Across all three evaluation stages, the key findings are:

- **Before SMOTE**: Logistic Regression leads on ROC AUC (82.73), showing a linear baseline is highly competitive on this four-feature space.
- **After SMOTE**: Random Forest benefits the most from oversampling and reaches the highest accuracy mean (80.77).
- **After Hyperparameter Tuning**: SVC and Logistic Regression deliver the best overall ROC AUC (0.75) and F1-score (0.61 / 0.60), while Random Forest holds the highest accuracy (0.75).

---

## How to Run

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/arigourumsah/Telco_Customer_Churn.git
cd Telco_Customer_Churn

# (Optional) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Notebook

```bash
jupyter notebook notebooks/telco_customer_churn_prediction.ipynb
```

Then run the cells sequentially to:

1. Load and understand the data.
2. Explore distributions, correlations, and churn patterns across features.
3. Engineer features, encode categorical variables, and select the top four predictors.
4. Split, scale, and run the cross-validated baseline benchmark.
5. Apply SMOTE oversampling and re-evaluate all models.
6. Tune top models with `GridSearchCV` and evaluate on the held-out test set.

> **Note:** If running in Google Colab, update the Drive mount path in the directory setup cell to match your own file location.

---

## Possible Extensions

Some ideas for extending this project:

- Experiment with a larger feature set beyond the top four chi-square features.
- Explore more advanced model interpretability tools (e.g., SHAP values) to explain individual churn predictions.
- Add more classifiers to the GridSearchCV tuning stage (e.g., XGBoost, LDA) for a broader comparison.
- Wrap the trained model into a simple REST API or Streamlit app for interactive churn scoring.
