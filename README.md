# Telco Customer Churn Prediction

A machine learning project that analyzes telecom customer behavior and predicts churn using exploratory data analysis, feature engineering, and classification models. The notebook works on a 7,043-row customer churn dataset with 21 columns and focuses on identifying the strongest churn drivers. 

## Project Overview

Customer churn is a critical problem for subscription businesses because lost customers directly affect revenue and retention costs. This project examines the patterns behind churn and builds predictive models to estimate whether a customer is likely to leave. 

The workflow in the notebook covers data loading, inspection, cleaning, feature selection, encoding, scaling, model training, and evaluation. The project also highlights the relationship between churn and contract type, tenure, and monthly charges. 

## Dataset

The dataset is stored in `data/raw/customer_churn_data.csv` and contains 7,043 records. The original dataset includes 21 columns such as `gender`, `SeniorCitizen`, `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`, and `Churn`. 

### Target variable

- `Churn`: `Yes` for customers who left and `No` for customers who stayed. The class distribution in the notebook is 1,869 churned customers and 5,174 retained customers. 

### Key fields used in modeling

- `tenure`.
- `Contract`.
- `MonthlyCharges`.
- `TotalCharges`. 

## Notebook Workflow

### 1. Data loading and inspection

The notebook loads the dataset from Google Drive, checks the shape and data types, and reviews summary statistics and sample rows. It also identifies that `TotalCharges` is stored as an object column and that the dataset contains a mix of numeric and categorical features. 

### 2. Exploratory data analysis

The analysis checks churn distribution and explores categorical churn patterns. The clearest relationship appears in `Contract`, where churn is highest for `Month-to-month` customers and lowest for `Two year` customers. 

The notebook also shows that churned customers have shorter average tenure than retained customers, with mean tenure around 17.98 for churned users versus 37.57 for retained users. 

### 3. Feature engineering

The target column `Churn` is mapped to binary values, `customerID` is removed, and `SeniorCitizen` is mapped to a more readable representation. Categorical variables are one-hot encoded, while `Contract` is manually ordinal-encoded as `Month-to-month = 0`, `One year = 1`, and `Two year = 2`. 

### 4. Feature selection

The notebook uses chi-square based feature selection and keeps `tenure`, `Contract`, `MonthlyCharges`, and `TotalCharges` as the selected predictors. This makes the final modeling stage more compact and focused on the strongest signals. 

### 5. Train-test split and scaling

The data is split into training and testing sets using an 80/20 split with stratification. Standard scaling is then applied to the selected features before model training. 

### 6. Model training and evaluation

Several classification models are trained and compared, including Logistic Regression, SVM, Decision Tree, Random Forest, KNN, Naive Bayes, LDA, and XGBoost. The notebook evaluates the models with cross-validated ROC AUC and accuracy. 

## Results

The best-performing model in the notebook is XGBoost, with a mean ROC AUC of 82.73 and a mean accuracy of 77.23. Logistic Regression also performs strongly, with a mean ROC AUC of 82.73 and a mean accuracy of 77.23, while the remaining models trail behind on the reported metrics. 

| Model | ROC AUC Mean | ROC AUC STD | Accuracy Mean | Accuracy STD |
|---|---:|---:|---:|---:|
| Logistic Regression | 82.73 | 1.97 | 77.23 | 2.20 |
| SVM | 81.04 | 2.12 | 70.57 | 2.46 |
| Decision Tree | 65.48 | 2.73 | 72.38 | 2.08 |
| Random Forest | 81.09 | 1.53 | 76.95 | 2.05 |
| KNN | 81.87 | 2.10 | 72.65 | 1.55 |
| Naive Bayes | 74.27 | 2.11 | 73.19 | 1.97 |
| LDA | 80.57 | 1.97 | 77.01 | 2.16 |
| XGBoost | 82.73 | 1.97 | 77.23 | 2.20 |

## Main Insights

- Customers on month-to-month contracts are far more likely to churn than customers on one-year or two-year contracts. 
- Shorter tenure is strongly associated with churn, which suggests that early retention efforts are important. 
- Monthly charges and total charges are useful predictors and were retained in the final feature set. 
- A compact model using only four selected features still produces competitive classification performance. 

## Tech Stack

- Python.
- pandas and numpy.
- matplotlib and seaborn.
- scikit-learn.
- XGBoost.
- imbalanced-learn.
- statsmodels. 

## Repository Structure

```bash
Telco_Customer_Churn/
├── data/
│   └── raw/
│       └── customer_churn_data.csv
├── models/
├── notebooks/
│   └── telco_customer_churn_prediction.ipynb.ipynb
├── README.md
└── requirements.txt
```

## How to Run

1. Clone the repository.
   ```bash
   git clone https://github.com/arigourumsah/Telco_Customer_Churn.git
   cd Telco_Customer_Churn
   ```

2. Create a virtual environment.
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies.
   ```bash
   pip install -r requirements.txt
   ```

4. Open the notebook.
   ```bash
   jupyter notebook
   ```

5. Run `notebooks/telco_customer_churn_prediction.ipynb.ipynb` from top to bottom.

## Business Value

This project can help telecom teams identify customers at risk of leaving and target them with retention offers before churn happens. The results also suggest that contract structure and customer tenure are especially important for churn reduction strategies. 

## Notes

The notebook is written in Google Colab and includes imported libraries for SMOTE, GridSearchCV, ROC-AUC analysis, and multiple classification algorithms. Some model tuning sections appear exploratory, so the README focuses on the validated workflow and reported results. 

## Author

**Bintang Arigo Kautsar Urumsah**

- GitHub: [arigourumsah](https://github.com/arigourumsah)
