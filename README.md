# 📊 Customer Churn Analysis

## 📌 Project Overview

This project focuses on analyzing customer churn data for a telecom company. The main goal is to understand customer characteristics, service details, contract types, payment methods, and monthly charges that can help in studying customer churn.

The project is being developed step by step using SQL and PostgreSQL, with the work maintained through Git and GitHub.

## 📂 Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains information about:

- 👥 Customer demographics
- 📄 Contract details
- 💳 Payment methods
- 📅 Tenure
- 💰 Monthly charges
- 👴 Senior citizen status
- 🔄 Churn status

## 🛠️ Tools Used

- PostgreSQL
- SQL
- Git
- GitHub

## 📅 week1 Day 1 - Data Setup and Basic Analysis

During Day 1, the dataset was loaded into PostgreSQL and basic checks were performed to understand the data.

### 🔎 Checks Performed

1. Total number of customers
2. Customer churn count
3. Customer count by gender
4. Customer count by contract type
5. Customer count by payment method
6. Customer count by senior citizen status
7. Minimum, maximum and average monthly charges

### 📈 Initial Results

- 👥 Total customers: **7,043**
- 💰 Average monthly charges: **64.76**
- ⬇️ Minimum monthly charges: **18.25**
- ⬆️ Maximum monthly charges: **118.75**
## 📅 Week 1 – Day 3 – Churn Analysis

During Day 3, customer churn was analyzed based on contract type and tenure to identify high-risk customer groups.

### 🔎 Analysis Performed

1. Churn by contract type
2. Churn by tenure
3. Comparison of churn patterns

### 📈 Results

- 📄 Month-to-month customers had the highest churn.
- 📅 Customers with shorter tenure showed higher churn.
- 🔍 Churn patterns were visualized for better understanding.

### 📁 Files Created

- `EDA.ipynb`

---

## 📅 Week 1 – Day 4 – Contract & Tenure Analysis

During Day 4, contract types and tenure groups were analyzed together to identify customers with the highest churn risk.

### 🔎 Analysis Performed

1. Contract-wise churn rate
2. Tenure group analysis
3. Contract and tenure comparison
4. Customer churn risk matrix

### 📈 Results

- 📄 Month-to-month customers had the highest churn rate.
- 📅 0–12 months tenure group had the highest churn.
- ⚠️ Month-to-month + 0–12 months was the highest-risk segment.
- 📊 Highest churn rate: **51.35%**

### 📁 Files Created

- `EDA.ipynb`

---

## 📅 Week 1 – Day 5 – Churn Driver Analysis

During Day 5, major factors influencing customer churn were analyzed to identify important churn drivers.

### 🔎 Analysis Performed

1. Payment method analysis
2. Internet service analysis
3. Monthly charges analysis
4. Additional service analysis

### 📈 Results

- 💳 Electronic check customers had the highest churn.
- 🌐 Fiber optic customers showed higher churn.
- 💰 Higher monthly charges were associated with higher churn.
- 🛠️ Customers without additional services showed higher churn.

### 📁 Files Created

- `EDA.ipynb`
- `Baseline_Analytics_Report.ipynb`

## 📂 Project Structure

```text
Customer-Churn-Prediction/
│
├── 01_Dataset_Telco-Customer-Churn.csv
│
├── 02_data_cleaning_eda.py
├── 03_prepare_ml_data.py
├── 04_train_logistic_regression.py
├── 05_train_random_forest.py
├── 06_train_xgboost.py
│
├── 06_xgboost_model.pkl
│
├── Week_2_Day_6_XGBoost.py
│
├── Week_3_Day_1_LTV_Regression.py
├── Week_3_Day_1_LTV_Visualization.py
├── Week_3_Day_1_LTV_Predictions.csv
├── Week_3_Day_1_Actual_vs_Predicted_LTV.png
├── Week_3_Day_1_LTV_Prediction_Error.png
│
├── Week_3_Day_2_LTV_Segmentation.py
├── Week_3_Day_2_LTV_Segmented_Customers.csv
├── Week_3_Day_2_01_LTV_Segment_Distribution.png
├── Week_3_Day_2_02_Average_LTV_by_Segment.png
│
├── Week_3_Day_4_FastAPI_App.py
├── Week_3_Day_5_Single_Customer_Inference.ipynb
├── Week_3_Day_6_7_FastAPI_Prediction_Service.ipynb
│
├── Week_4_Day_1_to_3_API_Integration.ipynb
├── Week_4_Day_1_to_3_Visualization_API_Integration.ipynb
├── Week_4_Day_3_Data_Validation.sql
│
├── dashboard.py
│
├── Dockerfile
├── requirements.txt
├── .dockerignore
│
└── README.md



## 📅 Week 1 – Day 6 – Data Cleaning & Feature Preparation

During Day 6, the dataset was prepared for Machine Learning by handling data quality issues and preparing categorical variables for further analysis.

### 🔎 Work Performed

1. Identified missing and inconsistent values.
2. Converted `TotalCharges` into a numeric data type.
3. Handled missing values in the dataset.
4. Reviewed categorical columns for Machine Learning preparation.
5. Prepared categorical variables for encoding.
6. Verified the cleaned dataset before feature engineering.

### 📈 Results

- 🧹 Data quality issues were identified and handled.
- 💰 `TotalCharges` was converted into a usable numeric format.
- 🔤 Categorical variables were prepared for encoding.
- ✅ The cleaned dataset was made ready for feature engineering and Machine Learning.

### 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- PostgreSQL
- SQL

### 📁 Files Created / Updated

- `02_data_cleaning_eda.py`
- `EDA.ipynb`

---
---

## 📅 Week 1 – Day 6 – Data Cleaning & Feature Preparation

During Day 6, the dataset was prepared for Machine Learning by handling data quality issues and preparing categorical variables for further analysis.

### 🔎 Work Performed

1. Identified missing and inconsistent values.
2. Converted `TotalCharges` into a numeric data type.
3. Handled missing values in the dataset.
4. Reviewed categorical columns for Machine Learning preparation.
5. Prepared categorical variables for encoding.
6. Verified the cleaned dataset before feature engineering.

### 📈 Results

- 🧹 Data quality issues were identified and handled.
- 💰 `TotalCharges` was converted into a usable numeric format.
- 🔤 Categorical variables were prepared for encoding.
- ✅ The cleaned dataset was made ready for feature engineering and Machine Learning.

### 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- PostgreSQL
- SQL

### 📁 Files Created / Updated

- `02_data_cleaning_eda.py`
- `EDA.ipynb`

---
# 📅 Week 2 – Feature Engineering & Predictive Modeling

## 📅 Week 2 – Day 1 – Machine Learning Data Preparation

During Day 1, the cleaned customer churn dataset was prepared for Machine Learning and feature engineering.

### 🔎 Work Performed

1. Loaded the cleaned customer churn dataset.
2. Selected relevant customer, service, contract, and billing features.
3. Identified numerical and categorical features.
4. Prepared the target variable for churn prediction.
5. Reviewed the dataset structure before feature engineering.
6. Prepared the data pipeline for subsequent Machine Learning models.

### 📈 Results

- 👥 Dataset records: **7,043 customers**
- 🎯 Churn was prepared as the target variable.
- 🔢 Numerical and categorical features were identified.
- 🤖 The dataset was prepared for feature engineering and predictive modeling.

### 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

### 📁 Files Created / Updated

- `03_prepare_ml_data.py`

### ✅ Day 1 Outcome

The cleaned customer dataset was successfully prepared as the foundation for **feature engineering and Machine Learning model development**.

---
---

## 📅 Week 2 – Day 2 – Customer Feature Engineering

During Day 2, additional customer-level features were created from the existing demographic, service, tenure, and billing information to improve the Machine Learning model's ability to identify churn patterns.

### 🔎 Features Engineered

1. **TenureGroup** – Customers grouped based on their tenure.
2. **TotalChargesPerTenure** – Total charges relative to customer tenure.
3. **MonthlyChargePerTenure** – Monthly charges relative to tenure.
4. **ServiceCount** – Number of subscribed services.
5. **MonthlyChargePerService** – Monthly charges relative to the number of services.
6. **TotalServiceCount** – Total number of subscribed services.
7. **MonthlyChargePerTotalService** – Monthly charges relative to total services.
8. **ServiceDensity** – Service usage relative to customer tenure.
9. **TotalChargesPerService** – Total charges relative to subscribed services.

### 📈 Results

- 🔢 New customer-level features were generated from the existing data.
- 📅 Tenure was converted into meaningful customer groups.
- 🛠️ Service-related features were created to represent customer engagement.
- 💰 Billing-related ratios were created to provide additional information about customer spending.
- 🤖 The engineered dataset was prepared for predictive modeling.

### 🛠️ Technologies Used

- Python
- Pandas
- NumPy

### 📁 Files Created / Updated

- `03_prepare_ml_data.py`
- Feature-engineered customer churn dataset

### ✅ Day 2 Outcome

Additional behavioral, service, tenure, and billing features were successfully engineered to improve the input data used for **Customer Churn Prediction**.

---
---

## 📅 Week 2 – Day 3 – Feature Engineering & ML Dataset Finalization

During Day 3, the engineered customer churn dataset was finalized and prepared for Machine Learning model training.

### 🔎 Work Performed

1. Finalized the engineered numerical and categorical features.
2. Converted the churn target into a numerical format.
3. Removed the `customerID` field from the Machine Learning input features.
4. Separated input features (`X`) and target variable (`y`).
5. Prepared numerical features for scaling.
6. Prepared categorical features for encoding.
7. Split the dataset into training and testing sets.
8. Verified the final feature dimensions and churn distribution.

### 📊 Dataset Preparation Results

- 👥 Total records: **7,043**
- 🔢 Features after encoding: **30**
- 🏋️ Training records: **5,634**
- 🧪 Testing records: **1,409**

### 🎯 Training Churn Distribution

- No Churn: **4,139**
- Churn: **1,495**

### 📈 Churn Distribution

- 🟢 No Churn: **73.46%**
- 🔴 Churn: **26.54%**

### 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn

### 📁 Files Created / Updated

- `03_prepare_ml_data.py`
- Feature-engineered ML dataset

### ✅ Day 3 Outcome

The final Machine Learning dataset was successfully prepared with engineered features, encoded categorical variables, and separate training and testing datasets.

The prepared data was ready for **classification model development from Week 2 Day 4 onwards**.

---
---

## 📅 Week 2 – Day 4 – Logistic Regression Model Training

During Day 4, a Logistic Regression classification model was trained to establish a baseline Machine Learning model for predicting customer churn.

### 🔎 Work Performed

1. Loaded the prepared Machine Learning dataset.
2. Separated input features and the churn target.
3. Applied preprocessing to numerical and categorical features.
4. Trained a Logistic Regression classifier.
5. Generated churn predictions on the test dataset.
6. Evaluated the model using classification metrics.

### 📊 Model Evaluation

The Logistic Regression model achieved a weighted average performance of approximately **0.80** across the evaluated classification metrics.

### ⚠️ Model Training Observation

A convergence warning was observed during training. The `max_iter` parameter was identified for adjustment to allow the model more iterations during optimization.

### 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Logistic Regression

### 📁 Files Created / Updated

- `04_train_logistic_regression.py`

### ✅ Day 4 Outcome

A baseline Logistic Regression model was successfully developed for customer churn prediction and its performance was evaluated as a reference for subsequent classification models.

---
---

## 📅 Week 2 – Day 5 – Random Forest Model Training

During Day 5, a Random Forest classification model was developed to improve customer churn prediction and compare its performance with the Logistic Regression baseline model.

### 🔎 Work Performed

1. Loaded the prepared Machine Learning dataset.
2. Separated input features and churn target.
3. Applied preprocessing to numerical and categorical features.
4. Trained a Random Forest classifier.
5. Generated churn predictions on the test dataset.
6. Evaluated the model using classification metrics.
7. Compared the model performance with the baseline Logistic Regression model.

### 📊 Model Evaluation

The Random Forest model was evaluated using:

- Accuracy
- Precision
- Recall
- F1-score

These metrics were used to understand the model's ability to correctly identify customers who were likely to churn.

### 🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest

### 📁 Files Created / Updated

- `05_train_random_forest.py`

### ✅ Day 5 Outcome

The Random Forest classification model was successfully trained and evaluated, providing an additional benchmark for comparison before developing the XGBoost churn prediction model.

---
---

## 📅 Week 2 – Day 6 – XGBoost Model Training

During Day 6, an XGBoost classification model was developed to predict customer churn using the engineered customer features.

### 🔎 Work Performed

1. Loaded the feature-engineered customer churn dataset.
2. Separated input features and the churn target.
3. Identified numerical and categorical features.
4. Applied preprocessing using:
   - StandardScaler for numerical features.
   - OneHotEncoder for categorical features.
5. Built an XGBoost classification model.
6. Trained the model using the prepared training dataset.
7. Generated churn predictions on the test dataset.
8. Evaluated the model using classification metrics.
9. Saved the trained XGBoost pipeline for later use in the dashboard.

### ⚙️ XGBoost Configuration

- Number of estimators: **300**
- Maximum depth: **4**
- Learning rate: **0.05**
- Subsample: **0.8**
- Column sampling: **0.8**
- Evaluation metric: **Log Loss**
- Random state: **42**

### 📊 Model Performance

| Metric | Score |
|---|---:|
| Accuracy | **79.13%** |
| Precision | **63.79%** |
| Recall | **49.47%** |
| F1-score | **55.72%** |

### 📁 Files Created / Updated

- `Week_2_Day_6_XGBoost.py`
- `06_xgboost_model.pkl`
- XGBoost prediction output

### ✅ Day 6 Outcome

The XGBoost churn classification model was successfully trained, evaluated, and saved as a reusable Machine Learning pipeline.

The saved model was later integrated into the **Streamlit Customer Churn Intelligence Dashboard** for real-time customer-level churn prediction.

---
---

## 📅 Week 2 – Day 7 – SHAP Explainable AI

During Day 7, SHAP (SHapley Additive exPlanations) was introduced to make the Machine Learning churn predictions more interpretable and understandable for business stakeholders.

### 🔎 Work Performed

1. Loaded the trained XGBoost model.
2. Prepared the required customer features for explanation.
3. Applied the same preprocessing used during model training.
4. Generated SHAP values for customer predictions.
5. Identified the features with the strongest influence on churn prediction.
6. Analyzed whether each feature increased or decreased churn risk.
7. Prepared the SHAP explanation for integration into the final dashboard.

### 🧠 SHAP Interpretation

- 🔴 Positive SHAP value → pushes the prediction toward higher churn risk.
- 🟢 Negative SHAP value → pushes the prediction toward lower churn risk.
- 📊 Larger absolute SHAP value → stronger influence on the prediction.

### 💡 Business Value

SHAP makes the churn prediction easier to understand by showing **why the model generated a particular prediction**, rather than displaying only the churn probability.

This allows business users to better understand customer risk factors and support targeted retention strategies.

### 🛠️ Technologies Used

- Python
- XGBoost
- SHAP
- Pandas
- Scikit-learn

### 📁 Files Created / Updated

- `Week_2_Day_6_XGBoost.py`
- `06_xgboost_model.pkl`

### ✅ Week 2 Outcome

The predictive modeling stage was completed with classification models and explainability preparation.

The project was ready to move into **Customer Lifetime Value (LTV) calculation and API development in Week 3**.

---
---

# 📅 Week 3 – LTV Calculation & API Development

## 📅 Week 3 – Day 1 – Customer Lifetime Value Prediction

During Day 1, a regression-based approach was developed to predict the expected Customer Lifetime Value (LTV) of customers.

### 🔎 Work Performed

1. Prepared the customer data required for LTV prediction.
2. Developed a regression model for predicting customer lifetime revenue.
3. Generated actual and predicted LTV values.
4. Calculated prediction errors.
5. Evaluated the LTV model using regression metrics.
6. Created visualizations to compare actual and predicted LTV values.

### 📊 Model Performance

| Metric | Value |
|---|---:|
| MAE | **50.43** |
| RMSE | **72.55** |
| R² Score | **0.9978** |

### 📈 LTV Results

- 💰 Mean Actual LTV: **$1,845.60**
- 💰 Mean Predicted LTV: **$1,841.00**
- 📊 Mean Prediction Error: **$4.61**
- ⚠️ Maximum Absolute Error: **$376.51**

### 📁 Files Created / Updated

- `Week_3_Day_1_LTV_Regression.py`
- `Week_3_Day_1_LTV_Visualization.py`
- `Week_3_Day_1_LTV_Predictions.csv`
- `Week_3_Day_1_01_Actual_vs_Predicted_LTV.png`
- `Week_3_Day_1_Actual_vs_Predicted_LTV.png`
- `Week_3_Day_1_LTV_Prediction_Error.png`

### ✅ Day 1 Outcome

The LTV prediction model was successfully developed and evaluated. Actual and predicted customer lifetime values were generated for further **LTV segmentation and business analysis**.

---
---

## 📅 Week 3 – Day 2 – Customer LTV Segmentation

During Day 2, the predicted Customer Lifetime Value results were analyzed and customers were segmented based on their LTV to support value-based customer prioritization.

### 🔎 Work Performed

1. Loaded the LTV prediction results.
2. Reviewed actual and predicted LTV values.
3. Calculated customer-level LTV segments.
4. Classified customers into different LTV categories.
5. Analyzed the distribution of customers across LTV segments.
6. Visualized the LTV segment distribution.
7. Compared average LTV across customer segments.

### 💎 LTV Segments

Customers were classified into:

- 🔴 **High LTV**
- 🟡 **Medium LTV**
- 🟢 **Low LTV**

### 📊 LTV Analysis

The segmentation helps identify customers based on their expected lifetime value and provides a foundation for prioritizing retention and marketing activities.

### 📈 Visualizations Created

- LTV Segment Distribution
- Average LTV by Segment

### 📁 Files Created / Updated

- `Week_3_Day_2_LTV_Segmentation.py`
- `Week_3_Day_2_LTV_Segmented_Customers.csv`
- `Week_3_Day_2_01_LTV_Segment_Distribution.png`
- `Week_3_Day_2_02_Average_LTV_by_Segment.png`

### 📊 Output Dataset

The final LTV segmentation dataset contains:

- **1,035 customer records**
- `Actual_LTV`
- `Predicted_LTV`
- `Prediction_Error`
- `LTV_Segment`

### ✅ Day 2 Outcome

Customer Lifetime Value segmentation was successfully completed, producing categorized LTV results for further **LTV analysis, visualization, and business decision-making**.

---
---

## 📅 Week 3 – Day 3 – LTV Model Evaluation

During Day 3, the LTV prediction model was evaluated in detail to measure its prediction accuracy and understand the difference between actual and predicted Customer Lifetime Value.

### 🔎 Work Performed

1. Evaluated the LTV regression model.
2. Calculated Mean Absolute Error (MAE).
3. Calculated Root Mean Squared Error (RMSE).
4. Calculated R² score.
5. Compared mean actual LTV with mean predicted LTV.
6. Analyzed prediction errors.
7. Identified the maximum absolute prediction error.

### 📊 Model Evaluation Results

| Metric | Value |
|---|---:|
| MAE | **$50.43** |
| RMSE | **$72.55** |
| R² Score | **0.9978** |

### 💰 LTV Summary

- 💵 Mean Actual LTV: **$1,845.60**
- 🤖 Mean Predicted LTV: **$1,841.00**
- 📊 Mean Prediction Error: **$4.61**
- ⚠️ Maximum Absolute Error: **$376.51**

### 📈 Key Observation

The actual and predicted average LTV values were close, with a mean prediction error of **$4.61**.

The evaluation results were used to assess the reliability of the LTV predictions before integrating the results into the later dashboard.

### 📁 Files Created / Updated

- `Week_3_Day_1_LTV_Regression.py`
- `Week_3_Day_1_LTV_Predictions.csv`

### ✅ Day 3 Outcome

The LTV prediction model was successfully evaluated using regression metrics and prediction-error analysis.

---
---

## 📅 Week 3 – Day 4 – FastAPI Application Development

During Day 4, a FastAPI application was developed to provide an API layer for the customer churn and prediction system.

### 🔎 Work Performed

1. Created the FastAPI application.
2. Configured the application to serve Machine Learning predictions.
3. Prepared the API structure for customer-level inference.
4. Integrated the prediction workflow with the application.
5. Configured the application for local API testing.
6. Prepared the API layer for further single-customer and batch prediction development.

### 🛠️ Technologies Used

- Python
- FastAPI
- Uvicorn
- Scikit-learn
- XGBoost

### 📁 Files Created / Updated

- `Week_3_Day_4_FastAPI_App.py`

### ✅ Day 4 Outcome

A FastAPI application was successfully created as the API layer of the Customer Churn Prediction system.

The application provided the foundation for **single-customer inference and prediction service development** in the following days.

---
---

## 📅 Week 3 – Day 5 – Single Customer Inference

During Day 5, single-customer inference was developed to test the prediction service using individual customer information.

### 🔎 Work Performed

1. Prepared individual customer input data.
2. Tested the prediction workflow for a single customer.
3. Passed customer features through the required preprocessing steps.
4. Generated a prediction for the selected customer.
5. Verified the single-customer inference workflow.
6. Prepared the inference process for integration with the prediction API.

### 🤖 Prediction Workflow

```text
Customer Input
      ↓
Data Preprocessing
      ↓
Feature Transformation
      ↓
Machine Learning Model
      ↓
Customer Prediction

---

## 📅 Week 3 – Day 6 & Day 7 – FastAPI Prediction Service

During Days 6 and 7, the FastAPI prediction service was developed and tested to provide Machine Learning predictions through an API-based workflow.

### 🔎 Work Performed

1. Integrated the trained Machine Learning model with the FastAPI service.
2. Prepared the API prediction workflow.
3. Implemented customer prediction processing.
4. Tested prediction requests through the API service.
5. Verified the model input and prediction output.
6. Prepared the service for further integration with the visualization layer.
7. Tested the FastAPI prediction workflow for reliable model inference.

### 🔄 Prediction Service Workflow

```text
Customer Data
      ↓
FastAPI Request
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
XGBoost Model
      ↓
Prediction
      ↓
API Response
---

# 📅 Week 4 – Visualization & Deployment

## 📅 Week 4 – Day 1 to Day 3 – API Integration, Visualization & Data Validation

During Days 1–3, the FastAPI prediction service and customer data were prepared for integration with the visualization layer. Data validation and API integration activities were performed before developing the final interactive dashboard.

### 🔎 Work Performed

1. Integrated the prediction API with the visualization workflow.
2. Prepared customer prediction results for visualization.
3. Tested the API-to-visualization workflow.
4. Validated the customer data used by the application.
5. Performed database/data validation checks.
6. Prepared the project components for interactive dashboard development.

### 🔄 Integration Workflow

```text
Customer Data
      ↓
Data Validation
      ↓
FastAPI Prediction Service
      ↓
Prediction Results
      ↓
Visualization Layer
      ↓
Interactive Dashboard

---

## 📅 Week 4 – Day 4 – Interactive Customer Churn Dashboard

During Day 4, an interactive Customer Churn Intelligence Dashboard was developed using Streamlit to bring together customer analytics and Machine Learning predictions in a single interface.

### 🔎 Work Performed

1. Developed the interactive Streamlit dashboard.
2. Added customer-level filtering options.
3. Created KPI cards for key churn metrics.
4. Integrated the trained XGBoost model for customer-level churn prediction.
5. Added churn probability calculation.
6. Implemented Low, Medium, and High risk classification.
7. Added retention recommendations based on predicted risk.
8. Created interactive churn analysis visualizations.
9. Added customer-level data exploration.
10. Prepared the dashboard for SHAP and LTV integration.

### 📊 Dashboard KPIs

- 👥 Total Customers
- 🔴 Churned Customers
- 📈 Churn Rate
- 💰 Average Monthly Charges
- 📅 Average Tenure

### 🎛️ Interactive Filters

Users can filter customers based on:

- Contract
- Tenure Group
- Internet Service
- Payment Method

### 🤖 AI Churn Prediction

The dashboard uses the trained XGBoost model to generate:

- Churn probability
- Predicted churn status
- Risk level
- Retention recommendation

### 🚦 Risk Classification

| Churn Probability | Risk Level |
|---|---|
| 0–39% | 🟢 Low |
| 40–69% | 🟡 Medium |
| 70–100% | 🔴 High |

The risk score represents the model's estimated probability and does not guarantee that a customer will churn.

### 📈 Visualizations

The dashboard includes:

- Churn by Contract
- Churn by Tenure
- Churn by Internet Service
- Churn by Payment Method
- Monthly Charges vs Tenure
- Churn Distribution

### 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- Plotly
- XGBoost
- Joblib

### 📁 Files Created / Updated

- `dashboard.py`
- `06_xgboost_model.pkl`

### ✅ Day 4 Outcome

An interactive **Customer Churn Intelligence Dashboard** was successfully developed, allowing users to explore churn patterns and generate customer-level Machine Learning predictions.

---
---

## 📅 Week 4 – Day 5 – SHAP Explainability & LTV Intelligence

During Day 5, advanced customer intelligence features were integrated into the Streamlit dashboard by combining SHAP-based explainability with Customer Lifetime Value (LTV) analysis.

### 🔎 Work Performed

1. Integrated SHAP Explainable AI into the dashboard.
2. Generated customer-level SHAP values.
3. Identified the features having the strongest influence on churn predictions.
4. Displayed whether individual features increased or reduced churn risk.
5. Integrated the existing LTV prediction results.
6. Integrated LTV customer segmentation.
7. Added LTV performance KPIs.
8. Added LTV segment distribution visualization.
9. Added average LTV by segment visualization.
10. Combined churn prediction and customer value analysis into one intelligence dashboard.

### 🧠 SHAP Explainability

The dashboard explains individual churn predictions using SHAP values.

- 🔴 Positive SHAP value → increases churn risk.
- 🟢 Negative SHAP value → reduces churn risk.
- 📊 Larger absolute SHAP value → stronger influence on the prediction.

### 💎 LTV Intelligence

The dashboard displays:

- Average Actual LTV
- Average Predicted LTV
- High-LTV Customers
- Total Customer LTV
- LTV Segment Distribution
- Average LTV by Segment

LTV monetary values are displayed in **USD ($)**.

### 📊 LTV Segments

Customers are categorized into:

- 🔴 High LTV
- 🟡 Medium LTV
- 🟢 Low LTV

### 💡 Business Value

The combination of **Churn Risk + Explainable AI + LTV Intelligence** helps identify customers who may require retention attention while also providing information about their customer value.

This supports more targeted and data-driven customer retention strategies.

### 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- XGBoost
- SHAP
- Joblib

### 📁 Files Created / Updated

- `dashboard.py`
- `06_xgboost_model.pkl`
- `Week_3_Day_2_LTV_Segmented_Customers.csv`

### ✅ Day 5 Outcome

SHAP Explainable AI and LTV Intelligence were successfully integrated into the interactive dashboard, completing the major customer-level intelligence features of the project.

---
---

## 📅 Week 4 – Day 6 – Docker Deployment

During Day 6, the Customer Churn Intelligence Dashboard was containerized using Docker to provide a consistent and portable deployment environment.

### 🔎 Work Performed

1. Installed and configured Docker Desktop.
2. Configured the Docker environment using the WSL 2 backend.
3. Created a `Dockerfile` for the Streamlit application.
4. Created `requirements.txt` containing the required Python dependencies.
5. Created `.dockerignore` to exclude unnecessary files from the Docker build context.
6. Built the Docker image for the dashboard.
7. Created and started a Docker container.
8. Configured port mapping for the Streamlit application.
9. Tested the dashboard through the Docker container.
10. Verified that the complete dashboard works successfully inside Docker.

### 🐳 Docker Configuration

**Docker Image:**

```text
customer-churn-dashboard
---

## 📅 Week 4 – Day 7 – Final Testing & Technical Documentation

Day 7 focused on final testing, documentation, project organization, and verification of the complete Customer Churn Prediction & LTV Engine.

### 🔎 Final Testing Performed

The complete Streamlit dashboard was tested after Docker deployment.

### 🧪 Dashboard Testing

The following features were verified successfully:

1. **Sidebar Filters**
   - Contract type
   - Tenure group
   - Internet service
   - Payment method

2. **Customer Churn KPIs**
   - Total customers
   - Churned customers
   - Churn rate
   - Average monthly charges
   - Average tenure

3. **AI Churn Prediction**
   - Customer information input
   - XGBoost-based prediction
   - Churn probability
   - Risk classification
   - Retention recommendation

4. **SHAP Explainability**
   - Individual customer selection
   - Top churn-driving features
   - Positive and negative feature contributions
   - Feature impact visualization

5. **LTV Intelligence**
   - Average actual LTV
   - Average predicted LTV
   - High-LTV customer count
   - Total customer LTV
   - LTV segment distribution
   - Average LTV by segment

6. **Docker Deployment**
   - Docker container running successfully
   - Streamlit application accessible through port `8501`
   - Dashboard features working inside the container

### 📚 Technical Documentation

The README was updated to document:

- Project overview
- Dataset
- Tools and technologies
- Week 1 development
- Week 2 feature engineering and ML modeling
- Week 3 LTV and API development
- Week 4 visualization and deployment
- Docker deployment
- Dashboard features
- SHAP explainability
- LTV intelligence

### ✅ Day 7 Outcome

The complete Customer Churn Prediction & LTV Engine was tested successfully.

The final system provides:

- 📊 Interactive customer churn analytics
- 🤖 AI-based churn prediction
- ⚠️ Customer risk classification
- 🔍 SHAP-based explainability
- 💰 LTV intelligence
- 📈 Interactive visualizations
- 🐳 Docker-based deployment
- 📚 Complete technical documentation

---