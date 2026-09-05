SELECT column_name, data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'telco_churn'
ORDER BY ordinal_position;


SELECT COUNT(*) AS total_customers
FROM public.telco_churn;

SELECT *
FROM public.telco_churn
LIMIT 5;

SELECT
    COUNT(*) AS total_customers,
    COUNT(DISTINCT customerid) AS unique_customers,
    COUNT(*) - COUNT(customerid) AS missing_customer_ids
FROM public.telco_churn;




SELECT
    churn,
    COUNT(*) AS customer_count
FROM public.telco_churn
GROUP BY churn
ORDER BY churn;


SELECT
    churn,
    COUNT(*) AS customer_count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) AS percentage
FROM public.telco_churn
GROUP BY churn
ORDER BY churn;



SELECT
    customerid,
    tenure,
    monthlycharges,
    totalcharges,
    churn
FROM public.telco_churn
LIMIT 10;




SELECT
    COUNT(*) AS total_rows,
    COUNT(tenure) AS tenure_non_null,
    COUNT(monthlycharges) AS monthlycharges_non_null,
    COUNT(totalcharges) AS totalcharges_non_null
FROM public.telco_churn;





SELECT
    COUNT(*) AS total_rows,
    COUNT(*) FILTER (WHERE TRIM(totalcharges) = '') AS blank_totalcharges,
    COUNT(*) FILTER (WHERE totalcharges IS NULL) AS null_totalcharges
FROM public.telco_churn;



SELECT
    customerid,
    tenure,
    monthlycharges,
    totalcharges
FROM public.telco_churn
WHERE TRIM(totalcharges) = ''
LIMIT 20;



SELECT
    churn,
    COUNT(*) AS blank_totalcharges_count
FROM public.telco_churn
WHERE TRIM(totalcharges) = ''
GROUP BY churn
ORDER BY churn;




SELECT
    COUNT(*) AS invalid_totalcharges
FROM public.telco_churn
WHERE NULLIF(TRIM(totalcharges), '') IS NOT NULL
  AND TRIM(totalcharges) !~ '^[0-9]+(\.[0-9]+)?$';



  SELECT
    customerid,
    tenure,
    monthlycharges,
    NULLIF(TRIM(totalcharges), '')::NUMERIC AS totalcharges_numeric,
    churn
FROM public.telco_churn
LIMIT 10;




SELECT
    customerid,
    tenure,
    monthlycharges,
    NULLIF(TRIM(totalcharges), '')::NUMERIC AS totalcharges_numeric,
    churn
FROM public.telco_churn
LIMIT 10;




SELECT
    COUNT(*) AS total_customers,
    COUNT(DISTINCT customerid) AS unique_customers,
    COUNT(*) FILTER (WHERE TRIM(totalcharges) = '') AS blank_totalcharges,
    COUNT(*) FILTER (WHERE churn = 'Yes') AS churned_customers,
    COUNT(*) FILTER (WHERE churn = 'No') AS active_customers
FROM public.telco_churn;

