-- =====================================================
-- Day 1–2: Data Setup & Basic SQL Analysis
-- Project: Customer Churn Analysis
-- Database: PostgreSQL
-- =====================================================


-- 1. Find Total Number of Customers
SELECT COUNT(*) AS total_records
FROM telco_churn;


-- 2. Find Customer Churn Count
SELECT churn, COUNT(*) AS customer_count
FROM telco_churn
GROUP BY churn;


-- 3. Find Customer Count by Gender
SELECT gender, COUNT(*) AS customer_count
FROM telco_churn
GROUP BY gender;


-- 4. Find Customer Count by Contract Type
SELECT contract, COUNT(*) AS customer_count
FROM telco_churn
GROUP BY contract;


-- 5. Find Customer Count by Payment Method
SELECT paymentmethod, COUNT(*) AS customer_count
FROM telco_churn
GROUP BY paymentmethod;


-- 6. Find Customer Count by Senior Citizen Status
SELECT seniorcitizen, COUNT(*) AS customer_count
FROM telco_churn
GROUP BY seniorcitizen;


-- 7. Find Minimum, Maximum and Average Monthly Charges
SELECT
    MIN(monthlycharges) AS min_monthly_charges,
    MAX(monthlycharges) AS max_monthly_charges,
    ROUND(AVG(monthlycharges), 2) AS avg_monthly_charges
FROM telco_churn;