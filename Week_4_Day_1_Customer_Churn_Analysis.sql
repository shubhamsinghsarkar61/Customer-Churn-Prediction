-- Week 4 Day 1: Basic Customer & Churn Analysis

SELECT
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    SUM(CASE WHEN churn = 'No' THEN 1 ELSE 0 END) AS retained_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_percent
FROM telco_churn;


-- Week 4 Day 1: Contract-wise Churn Analysis

SELECT
    contract,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_percent
FROM telco_churn
GROUP BY contract
ORDER BY churn_rate_percent DESC;






-- Week 4 Day 1: Tenure-wise Churn Analysis

SELECT
    CASE
        WHEN tenure < 12 THEN '0-11 Months'
        WHEN tenure < 24 THEN '12-23 Months'
        WHEN tenure < 48 THEN '24-47 Months'
        ELSE '48+ Months'
    END AS tenure_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_percent
FROM telco_churn
GROUP BY tenure_group
ORDER BY churn_rate_percent DESC;






-- Week 4 Day 1: Monthly Charges-wise Churn Analysis

SELECT
    CASE
        WHEN monthlycharges < 30 THEN 'Under 30'
        WHEN monthlycharges < 60 THEN '30-59'
        WHEN monthlycharges < 90 THEN '60-89'
        ELSE '90+'
    END AS monthly_charge_group,
    COUNT(*) AS total_customers,
    SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) AS churned_customers,
    ROUND(
        100.0 * SUM(CASE WHEN churn = 'Yes' THEN 1 ELSE 0 END) / COUNT(*),
        2
    ) AS churn_rate_percent
FROM telco_churn
GROUP BY monthly_charge_group
ORDER BY churn_rate_percent DESC;




-- ============================================
-- Week 4 Day 1: Business Insights Summary
-- ============================================

-- Key Findings:
-- 1. Overall customer churn rate is 26.54%.
-- 2. Month-to-month customers have the highest contract-wise churn rate (42.71%).
-- 3. Customers with 0-11 months tenure have the highest tenure-wise churn rate (48.28%).
-- 4. Customers with monthly charges between 60-89 have the highest churn rate (33.74%).
-- 5. Churn generally decreases as customer tenure increases.sss