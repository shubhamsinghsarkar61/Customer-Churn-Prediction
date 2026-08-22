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

## 📁 Project Structure

```text
Customer-Churn-Prediction/
│
├── WA_Fn-UseC_-Telco-Customer-Churn.csv
└── README.md
