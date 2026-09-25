# Telco Customer Churn Prediction

A machine learning project that analyzes telecom customer behavior and predicts churn using exploratory data analysis, feature engineering, and classification models. The notebook works on a 7,043-row customer churn dataset with 21 columns and focuses on identifying the strongest churn drivers. 

## Project Overview

Customer churn is a critical problem for subscription businesses because lost customers directly affect revenue and retention costs. This project examines the patterns behind churn and builds predictive models to estimate whether a customer is likely to leave. 

The workflow in the notebook covers data loading, inspection, cleaning, feature selection, encoding, scaling, model training, and evaluation. The project also highlights the relationship between churn and contract type, tenure, and monthly charges. 

## Dataset

I use `data/raw/customer_churn_data.csv` as the source dataset for this project. The dataset contains 7,043 rows and 21 columns, including customer demographics, service usage, account details, billing features, and the churn label.

The raw columns in the notebook are:

- `customerID`
- `gender`
- `SeniorCitizen`
- `Partner`
- `Dependents`
- `tenure`
- `PhoneService`
- `MultipleLines`
- `InternetService`
- `OnlineSecurity`
- `OnlineBackup`
- `DeviceProtection`
- `TechSupport`
- `StreamingTV`
- `StreamingMovies`
- `Contract`
- `PaperlessBilling`
- `PaymentMethod`
- `MonthlyCharges`
- `TotalCharges`
- `Churn`

### Target variable

- `Churn`: `Yes` for customers who left and `No` for customers who stayed. The class distribution in the notebook is 1,869 churned customers and 5,174 retained customers. 

### Key fields used in modeling

- `tenure`.
- `Contract`.
- `MonthlyCharges`.
- `TotalCharges`. 

## Notebook Workflow

### 1. Data Cleaning and Handling

I begin by loading the dataset and checking its structure with `info()`, `head()`, `describe()`, and full table inspection. At this stage, I confirm that the dataset has 7,043 entries. I inspect basic numerical summaries for variables such as `SeniorCitizen`, `tenure`, and `MonthlyCharges`. For example, the notebook shows an average tenure of 32.37 and an average monthly charge of 64.76 in the raw dataset. I also observe that `TotalCharges` is loaded as an object column instead of a numeric column. This is due to some records containing strings with only whitespaces or " ". Since these records also show their `tenure = 0`, meaning they are new customers, I replaced these empty strings with 0, and then converted the column to a numerical data type. Additionally, There are multiple categorical columns (`OnlineSecurity, DeviceProtection, TechSupport, StreamingTV, StreamingMovies`) that have an unnecessary category `No internet service` and one column (`MultipleLines`) with the category `No Phone service`. These columns also contained categories `Yes` and `No`. Thus, I converted the unnecessary categories to `No` to standardize these columns.

### 2. Exploratory Data Analysis (EDA)

I continue with exploratory data analysis by reviewing the distribution of the raw variables, checking correlations for the numerical features, and studying churn behavior across categorical and numerical variables. To make this easier to read, I group the EDA into five visual analysis blocks that follow the notebook order. 

#### 2.1. Data Distribution

![Data Distribution of the numerical features](/images/data_distribution.png)

In the data distribution plots, I inspect the spread of the main numerical features: `SeniorCitizen`, `tenure`, `MonthlyCharges`, and `TotalCharges`. `SeniorCitizen` is highly imbalanced because most customers are non-seniors, `tenure` spans the full service range with visible concentration at lower and higher values, `MonthlyCharges` shows a broad spread across service plans, and `TotalCharges` is right-skewed because many customers have low accumulated billing while fewer customers have very high totals. 

This visualization helps me understand the overall shape of the data before modeling. It also confirms that the dataset mixes binary, discrete, and continuous-like variables, which is important for choosing later preprocessing steps. 

#### 2.2. Boxplot Analysis

![Boxplot Analysis of the numerical features against the Churn label](/images/boxplots.png)

In the boxplots, I compare `tenure`, `MonthlyCharges`, and `TotalCharges` against the churn label. The `tenure` boxplot shows the strongest separation: churned customers have a much lower median tenure and a tighter concentration near the beginning of the customer lifecycle, while retained customers have a much wider tenure distribution. 

The `MonthlyCharges` boxplot shows that churned customers tend to sit at higher charge levels than non-churned customers. The `TotalCharges` boxplot also shows a clear difference, with retained customers generally accumulating much larger total charges because they stay longer, even though churned customers still have a range of high outliers. 

#### 2.3. Correlation Heatmap

![Correlation Heatmap of the numerical features with Churn label](/images/correlation_heatmap.png)

The correlation heatmap focuses on `tenure`, `MonthlyCharges`, `TotalCharges`, and `Churn`. I see a strong positive correlation of 0.83 between `tenure` and `TotalCharges`, a moderate positive correlation of 0.65 between `MonthlyCharges` and `TotalCharges`, and a negative correlation of -0.35 between `tenure` and `Churn`. 

`MonthlyCharges` and `Churn` show a weaker positive relationship of 0.19, while `TotalCharges` and `Churn` show a negative relationship of -0.20. This tells me that tenure is the strongest linear signal in the set, while billing variables contribute additional but weaker churn information. 

#### 2.4. Countplots of Categorical Features

![Countplot Analysis of the categorical features](/images/countplots.png)

I then visualize the raw frequency distribution of the categorical variables using countplots. These plots show that the dataset is fairly balanced by `gender`, slightly skewed toward customers without a `Partner`, and more skewed toward customers without `Dependents`. 

Several service-related variables also show clear usage patterns. Most customers have `PhoneService`, many have `MultipleLines = No`, `InternetService` is dominated by `Fiber optic` and `DSL`, and the availability of add-on services such as `OnlineSecurity`, `DeviceProtection`, `TechSupport`, `StreamingTV`, and `StreamingMovies` varies noticeably across the customer base. 

The `Contract` countplot is especially important because it shows that `Month-to-month` is the most common contract type, followed by `One year`, then `Two year`. `PaperlessBilling` also shows a notable skew toward `Yes`, and `PaymentMethod` is spread across several payment options with `Electronic check` appearing prominently. 

#### 2.5. Proportion of Churn in Categorical Features

![Proportion Analysis of the categorical features](/images/proportions.png)

After the raw countplots, I plot stacked proportion charts to see how churn is distributed inside each category. This is more informative than counts alone because it shows whether a category has a higher churn share even when its total population is larger. 

The clearest pattern appears in `Contract`: `Month-to-month` has the largest churn proportion, while `One year` and especially `Two year` have much smaller churn shares. This reinforces the idea that longer commitments are associated with retention.

I also see elevated churn proportions for customers without `OnlineSecurity`, `TechSupport`, and `DeviceProtection`, and for customers using `Fiber optic` internet service. `PaperlessBilling` and `PaymentMethod` also show meaningful churn differences, with `Electronic check` standing out as a higher-risk payment method in the stacked proportion plot. 

### 3. Feature engineering

After EDA, I move to feature engineering. I first drop `customerID`, and remap `SeniorCitizen` from `1/0` into `Yes/No` so it can be handled consistently with the other categorical variables.

Next, I apply one-hot encoding to the categorical variables, including `gender`, `SeniorCitizen`, `Partner`, `Dependents`, `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`, `PaperlessBilling`, and `PaymentMethod`.

For `Contract`, I keep a manual ordinal representation instead of one-hot encoding it. In the notebook, I map `Month-to-month = 0`, `One year = 1`, and `Two year = 2`.

### 4. Feature selection

Once the feature matrix is prepared, I run chi-square feature selection using `SelectKBest(chi2, k=4)`. The selected predictors are `tenure`, `Contract`, `MonthlyCharges`, and `TotalCharges`.

This gives me a smaller and more focused feature set for the downstream models. I then define `X_new` using those four selected variables and keep `Churn` as the target.

### 5. Train Test Split

After selecting the final features, I split the data into training and testing sets using `train_test_split` with `test_size = 0.2`, `random_state = 42`, and `stratify = y`. This keeps the class balance more stable between the train and test partitions. 

### 6. Scaling Train and Test Dataset

Next, I standardize the selected numerical features using `StandardScaler`. I fit the scaler on the training data and apply the same transformation to the test data so both sets stay on the same scale.

### 7. Model training and evaluation

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

5. Run `notebooks/telco_customer_churn_prediction.ipynb` from top to bottom.

## Business Value

This project can help telecom teams identify customers at risk of leaving and target them with retention offers before churn happens. The results also suggest that contract structure and customer tenure are especially important for churn reduction strategies. 

## Notes

The notebook is written in Google Colab and includes imported libraries for SMOTE, GridSearchCV, ROC-AUC analysis, and multiple classification algorithms. Some model tuning sections appear exploratory, so the README focuses on the validated workflow and reported results. 

## Author

**Bintang Arigo Kautsar Urumsah**

- GitHub: [arigourumsah](https://github.com/arigourumsah)
