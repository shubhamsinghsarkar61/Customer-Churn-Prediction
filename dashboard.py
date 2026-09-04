import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

h1 {
    font-weight: 700;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATA
# =========================================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "01_Dataset_Telco-Customer-Churn.csv"
    )

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing TotalCharges
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Create Tenure Group
    def tenure_group(tenure):

        if tenure <= 12:
            return "0–12 Months"

        elif tenure <= 24:
            return "13–24 Months"

        elif tenure <= 48:
            return "25–48 Months"

        else:
            return "49–72 Months"

    df["Tenure Group"] = df["tenure"].apply(
        tenure_group
    )

    return df


df = load_data()


# =========================================================
# LOAD XGBOOST MODEL
# =========================================================

@st.cache_resource
def load_model():

    try:

        model = joblib.load(
            "06_xgboost_model.pkl"
        )

        return model

    except FileNotFoundError:

        return None


model = load_model()


# =========================================================
# TITLE
# =========================================================

st.title(
    "📊 Customer Churn Intelligence Dashboard"
)

st.markdown(
    "### Interactive analytics + AI-powered churn prediction"
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header(
    "🔍 Dashboard Filters"
)


# ---------------------------------------------------------
# CONTRACT
# ---------------------------------------------------------

contract_options = [
    "All"
] + sorted(
    df["Contract"]
    .dropna()
    .unique()
    .tolist()
)

contract_filter = st.sidebar.selectbox(
    "Contract",
    contract_options
)


# ---------------------------------------------------------
# TENURE GROUP
# ---------------------------------------------------------

tenure_options = [
    "All",
    "0–12 Months",
    "13–24 Months",
    "25–48 Months",
    "49–72 Months"
]

tenure_filter = st.sidebar.selectbox(
    "Tenure Group",
    tenure_options
)


# ---------------------------------------------------------
# INTERNET SERVICE
# ---------------------------------------------------------

internet_options = [
    "All"
] + sorted(
    df["InternetService"]
    .dropna()
    .unique()
    .tolist()
)

internet_filter = st.sidebar.selectbox(
    "Internet Service",
    internet_options
)


# ---------------------------------------------------------
# PAYMENT METHOD
# ---------------------------------------------------------

payment_options = [
    "All"
] + sorted(
    df["PaymentMethod"]
    .dropna()
    .unique()
    .tolist()
)

payment_filter = st.sidebar.selectbox(
    "Payment Method",
    payment_options
)


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df.copy()


if contract_filter != "All":

    filtered_df = filtered_df[
        filtered_df["Contract"]
        == contract_filter
    ]


if tenure_filter != "All":

    filtered_df = filtered_df[
        filtered_df["Tenure Group"]
        == tenure_filter
    ]


if internet_filter != "All":

    filtered_df = filtered_df[
        filtered_df["InternetService"]
        == internet_filter
    ]


if payment_filter != "All":

    filtered_df = filtered_df[
        filtered_df["PaymentMethod"]
        == payment_filter
    ]


# =========================================================
# KPI CALCULATIONS
# =========================================================

total_customers = len(
    filtered_df
)

churned_customers = (
    filtered_df["Churn"] == "Yes"
).sum()


if total_customers > 0:

    churn_rate = (
        churned_customers
        / total_customers
    ) * 100

else:

    churn_rate = 0


if total_customers > 0:

    avg_monthly = (
        filtered_df["MonthlyCharges"]
        .mean()
    )

else:

    avg_monthly = 0


if total_customers > 0:

    avg_tenure = (
        filtered_df["tenure"]
        .mean()
    )

else:

    avg_tenure = 0


# =========================================================
# EXECUTIVE OVERVIEW
# =========================================================

st.subheader(
    "📌 Executive Overview"
)


col1, col2, col3, col4, col5 = st.columns(5)


col1.metric(
    "Total Customers",
    f"{total_customers:,}"
)


col2.metric(
    "Churned Customers",
    f"{churned_customers:,}"
)


col3.metric(
    "Churn Rate",
    f"{churn_rate:.2f}%"
)


col4.metric(
    "Avg Monthly Charges",
    f"${avg_monthly:.2f}"
)


col5.metric(
    "Avg Tenure",
    f"{avg_tenure:.1f} Months"
)


st.divider()


# =========================================================
# AI CHURN PREDICTION
# =========================================================

st.header(
    "🤖 AI Churn Prediction"
)

st.markdown(
    "Enter customer information to estimate the customer's churn probability using the trained XGBoost model."
)


if model is None:

    st.error(
        "❌ XGBoost model file not found. "
        "Make sure 06_xgboost_model.pkl is in the project folder."
    )

else:

    st.success(
        "✅ XGBoost model loaded successfully."
    )


    # -----------------------------------------------------
    # CUSTOMER INPUT
    # -----------------------------------------------------

    pred_col1, pred_col2, pred_col3 = st.columns(3)


    with pred_col1:

        gender = st.selectbox(
            "Gender",
            sorted(
                df["gender"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No"
        )

        partner = st.selectbox(
            "Partner",
            sorted(
                df["Partner"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        dependents = st.selectbox(
            "Dependents",
            sorted(
                df["Dependents"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with pred_col2:

        tenure = st.slider(
            "Tenure (Months)",
            min_value=0,
            max_value=72,
            value=12
        )

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            max_value=200.0,
            value=70.0,
            step=1.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            max_value=10000.0,
            value=840.0,
            step=10.0
        )

        contract = st.selectbox(
            "Contract",
            sorted(
                df["Contract"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with pred_col3:

        phone_service = st.selectbox(
            "Phone Service",
            sorted(
                df["PhoneService"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        multiple_lines = st.selectbox(
            "Multiple Lines",
            sorted(
                df["MultipleLines"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        internet_service = st.selectbox(
            "Internet Service",
            sorted(
                df["InternetService"]
                .dropna()
                .unique()
                .tolist()
            )
        )

        payment_method = st.selectbox(
            "Payment Method",
            sorted(
                df["PaymentMethod"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    # -----------------------------------------------------
    # ADDITIONAL SERVICES
    # -----------------------------------------------------

    st.subheader(
        "🛠️ Customer Services"
    )


    service_col1, service_col2, service_col3, service_col4 = st.columns(4)


    with service_col1:

        online_security = st.selectbox(
            "Online Security",
            sorted(
                df["OnlineSecurity"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with service_col2:

        online_backup = st.selectbox(
            "Online Backup",
            sorted(
                df["OnlineBackup"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with service_col3:

        device_protection = st.selectbox(
            "Device Protection",
            sorted(
                df["DeviceProtection"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with service_col4:

        tech_support = st.selectbox(
            "Tech Support",
            sorted(
                df["TechSupport"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    service_col5, service_col6, service_col7, service_col8 = st.columns(4)


    with service_col5:

        streaming_tv = st.selectbox(
            "Streaming TV",
            sorted(
                df["StreamingTV"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with service_col6:

        streaming_movies = st.selectbox(
            "Streaming Movies",
            sorted(
                df["StreamingMovies"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with service_col7:

        paperless_billing = st.selectbox(
            "Paperless Billing",
            sorted(
                df["PaperlessBilling"]
                .dropna()
                .unique()
                .tolist()
            )
        )


    with service_col8:

        st.write("")
        st.write("")
        predict_button = st.button(
            "🔮 Predict Churn",
            width="stretch"
        )


    # -----------------------------------------------------
    # PREDICTION
    # -----------------------------------------------------

    if predict_button:

        # Create tenure group
        if tenure <= 12:

            tenure_group_value = "0–12 Months"

        elif tenure <= 24:

            tenure_group_value = "13–24 Months"

        elif tenure <= 48:

            tenure_group_value = "25–48 Months"

        else:

            tenure_group_value = "49–72 Months"


        # Feature engineering
        if tenure > 0:

            total_charges_per_tenure = (
                total_charges / tenure
            )

            monthly_charge_per_tenure = (
                monthly_charges / tenure
            )

        else:

            total_charges_per_tenure = 0

            monthly_charge_per_tenure = 0


        # Service count
        service_values = [
            phone_service,
            multiple_lines,
            internet_service,
            online_security,
            online_backup,
            device_protection,
            tech_support,
            streaming_tv,
            streaming_movies
        ]


        service_count = sum(
            1
            for value in service_values
            if value not in ["No", "No internet service"]
        )


        total_service_count = (
            service_count + 1
        )


        if service_count > 0:

            monthly_charge_per_service = (
                monthly_charges
                / service_count
            )

        else:

            monthly_charge_per_service = 0


        if total_service_count > 0:

            monthly_charge_per_total_service = (
                monthly_charges
                / total_service_count
            )

        else:

            monthly_charge_per_total_service = 0


        if tenure > 0:

            service_density = (
                service_count
                / tenure
            )

        else:

            service_density = 0


        if service_count > 0:

            total_charges_per_service = (
                total_charges
                / service_count
            )

        else:

            total_charges_per_service = 0


        # -------------------------------------------------
        # CREATE INPUT DATAFRAME
        # -------------------------------------------------

        prediction_data = pd.DataFrame({

            "gender": [gender],

            "SeniorCitizen": [
                senior_citizen
            ],

            "Partner": [partner],

            "Dependents": [
                dependents
            ],

            "tenure": [tenure],

            "PhoneService": [
                phone_service
            ],

            "MultipleLines": [
                multiple_lines
            ],

            "InternetService": [
                internet_service
            ],

            "OnlineSecurity": [
                online_security
            ],

            "OnlineBackup": [
                online_backup
            ],

            "DeviceProtection": [
                device_protection
            ],

            "TechSupport": [
                tech_support
            ],

            "StreamingTV": [
                streaming_tv
            ],

            "StreamingMovies": [
                streaming_movies
            ],

            "Contract": [
                contract
            ],

            "PaperlessBilling": [
                paperless_billing
            ],

            "PaymentMethod": [
                payment_method
            ],

            "MonthlyCharges": [
                monthly_charges
            ],

            "TotalCharges": [
                total_charges
            ],

            "TotalChargesPerTenure": [
                total_charges_per_tenure
            ],

            "MonthlyChargePerTenure": [
                monthly_charge_per_tenure
            ],

            "ServiceCount": [
                service_count
            ],

            "MonthlyChargePerService": [
                monthly_charge_per_service
            ],

            "TotalServiceCount": [
                total_service_count
            ],

            "MonthlyChargePerTotalService": [
                monthly_charge_per_total_service
            ],

            "ServiceDensity": [
                service_density
            ],

            "TotalChargesPerService": [
                total_charges_per_service
            ],

            "TenureGroup": [
                tenure_group_value
            ]
        })

        try:

            # Prediction probability
            probability = model.predict_proba(
                prediction_data
            )[0][1]


            prediction = model.predict(
                prediction_data
            )[0]


            probability_percent = (
                probability * 100
            )


            # -------------------------------------------------
            # RISK LEVEL
            # -------------------------------------------------

            if probability_percent >= 70:

                risk_level = "HIGH"

            elif probability_percent >= 40:

                risk_level = "MEDIUM"

            else:

                risk_level = "LOW"

# -------------------------------------------------
# VISUAL PREDICTION RESULTS
# -------------------------------------------------
        except Exception as e:

            st.error(
                "Prediction could not be completed."
            )

            st.exception(e)
st.divider()

st.subheader(
    "🎯 Prediction Result"
)

# =================================================
# CHURN PROBABILITY GAUGE
# =================================================

fig_gauge = go.Figure(
    go.Indicator(
        mode="gauge+number",
        value=probability_percent,

        number={
            "suffix": "%",
            "font": {
                "size": 40
            }
        },

        title={
            "text": "Churn Probability"
        },

        gauge={
            "axis": {
                "range": [0, 100]
            },

            "bar": {
                "color": (
                    "#e74c3c"
                    if probability_percent >= 70
                    else
                    "#f1c40f"
                    if probability_percent >= 40
                    else
                    "#2ecc71"
                )
            },

            "steps": [
                {
                    "range": [0, 40],
                    "color": "#d5f5e3"
                },

                {
                    "range": [40, 70],
                    "color": "#fcf3cf"
                },

                {
                    "range": [70, 100],
                    "color": "#fadbd8"
                }
            ],

            "threshold": {
                "line": {
                    "color": "black",
                    "width": 4
                },

                "thickness": 0.75,

                "value": 70
            }
        }
    )
)

st.plotly_chart(
    fig_gauge,
    width="stretch"
)


# =================================================
# PREDICTION STATUS
# =================================================

if risk_level == "HIGH":

    st.error(
        f"🔴 HIGH CHURN RISK — "
        f"{probability_percent:.2f}%"
    )

    st.markdown(
        """
        **Prediction:** Customer is likely to churn.

        **Recommended Action:** Immediate retention
        intervention, personalized offer, and
        customer support follow-up.
        """
    )

elif risk_level == "MEDIUM":

    st.warning(
        f"🟡 MEDIUM CHURN RISK — "
        f"{probability_percent:.2f}%"
    )

    st.markdown(
        """
        **Prediction:** Customer is at risk of churn.

        **Recommended Action:** Proactive engagement
        and targeted retention offers.
        """
    )

else:

    st.success(
        f"🟢 LOW CHURN RISK — "
        f"{probability_percent:.2f}%"
    )

    st.markdown(
        """
        **Prediction:** Customer is likely to stay.

        **Recommended Action:** Continue regular
        engagement and maintain service quality.
        """
    )


# =================================================
# PREDICTION PROBABILITY BAR
# =================================================

prediction_chart = pd.DataFrame({

    "Status": [
        "Churn Probability",
        "Stay Probability"
    ],

    "Percentage": [
        probability_percent,
        100 - probability_percent
    ]
})


fig_prediction = px.bar(
    prediction_chart,

    x="Status",

    y="Percentage",

    text="Percentage",

    range_y=[0, 100],

    title="Churn vs Stay Probability"
)


fig_prediction.update_traces(
    texttemplate="%{text:.1f}%",

    textposition="outside"
)


fig_prediction.update_layout(

    yaxis_title="Probability (%)",

    xaxis_title="",

    showlegend=False
)


st.plotly_chart(
    fig_prediction,
    width="stretch"
)


# =================================================
# RETENTION RECOMMENDATION
# =================================================

st.subheader(
    "💡 Retention Recommendation"
)

if risk_level == "HIGH":

    st.error(
        "🚨 High-risk customer. "
        "Consider immediate retention action, "
        "personalized offers, service support, "
        "or contract incentives."
    )

elif risk_level == "MEDIUM":

    st.warning(
        "⚠️ Medium-risk customer. "
        "Consider proactive engagement, "
        "service improvements, and targeted offers."
    )

else:

    st.success(
        "✅ Low-risk customer. "
        "Continue regular engagement and "
        "maintain service quality."
    )

# =========================================================
# CHURN BY CONTRACT
# =========================================================

st.subheader(
    "📄 Churn Rate by Contract"
)


contract_churn = (
    filtered_df
    .groupby("Contract")["Churn"]
    .apply(
        lambda x:
        (x == "Yes").mean() * 100
    )
    .reset_index(
        name="Churn Rate"
    )
)


fig_contract = px.bar(
    contract_churn,
    x="Contract",
    y="Churn Rate",
    text="Churn Rate",
    title="Churn Rate by Contract"
)


fig_contract.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)


fig_contract.update_layout(
    yaxis_title="Churn Rate (%)",
    xaxis_title="Contract",
    yaxis_range=[0, 100]
)


st.plotly_chart(
    fig_contract,
    width="stretch"
)


# =========================================================
# TWO COLUMN SECTION
# =========================================================

col1, col2 = st.columns(2)


# =========================================================
# CHURN BY TENURE
# =========================================================

with col1:

    st.subheader(
        "⏳ Churn Rate by Tenure"
    )


    tenure_order = [
        "0–12 Months",
        "13–24 Months",
        "25–48 Months",
        "49–72 Months"
    ]


    tenure_churn = (
        filtered_df
        .groupby("Tenure Group")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="Churn Rate"
        )
    )


    tenure_churn["Tenure Group"] = pd.Categorical(
        tenure_churn["Tenure Group"],
        categories=tenure_order,
        ordered=True
    )


    tenure_churn = tenure_churn.sort_values(
        "Tenure Group"
    )


    fig_tenure = px.bar(
        tenure_churn,
        x="Tenure Group",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Tenure Group"
    )


    fig_tenure.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig_tenure.update_layout(
        yaxis_title="Churn Rate (%)",
        xaxis_title="Tenure Group",
        yaxis_range=[0, 100]
    )


    st.plotly_chart(
        fig_tenure,
        width="stretch"
    )


# =========================================================
# INTERNET SERVICE
# =========================================================

with col2:

    st.subheader(
        "🌐 Churn Rate by Internet Service"
    )


    internet_churn = (
        filtered_df
        .groupby("InternetService")["Churn"]
        .apply(
            lambda x:
            (x == "Yes").mean() * 100
        )
        .reset_index(
            name="Churn Rate"
        )
    )


    fig_internet = px.bar(
        internet_churn,
        x="InternetService",
        y="Churn Rate",
        text="Churn Rate",
        title="Churn Rate by Internet Service"
    )


    fig_internet.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside"
    )


    fig_internet.update_layout(
        yaxis_title="Churn Rate (%)",
        xaxis_title="Internet Service",
        yaxis_range=[0, 100]
    )


    st.plotly_chart(
        fig_internet,
        width="stretch"
    )


# =========================================================
# PAYMENT METHOD
# =========================================================

st.subheader(
    "💳 Churn Rate by Payment Method"
)


payment_churn = (
    filtered_df
    .groupby("PaymentMethod")["Churn"]
    .apply(
        lambda x:
        (x == "Yes").mean() * 100
    )
    .reset_index(
        name="Churn Rate"
    )
)


fig_payment = px.bar(
    payment_churn,
    x="PaymentMethod",
    y="Churn Rate",
    text="Churn Rate",
    title="Churn Rate by Payment Method"
)


fig_payment.update_traces(
    texttemplate="%{text:.2f}%",
    textposition="outside"
)


fig_payment.update_layout(
    yaxis_title="Churn Rate (%)",
    xaxis_title="Payment Method",
    yaxis_range=[0, 100]
)


st.plotly_chart(
    fig_payment,
    width="stretch"
)


# =========================================================
# MONTHLY CHARGES VS TENURE
# =========================================================

st.subheader(
    "💰 Monthly Charges vs Tenure"
)


fig_scatter = px.scatter(
    filtered_df,
    x="tenure",
    y="MonthlyCharges",
    color="Churn",
    hover_data=[
        "customerID",
        "Contract",
        "InternetService",
        "PaymentMethod"
    ],
    title="Customer Charges vs Tenure"
)


fig_scatter.update_layout(
    xaxis_title="Tenure (Months)",
    yaxis_title="Monthly Charges"
)


st.plotly_chart(
    fig_scatter,
    width="stretch"
)


# =========================================================
# CUSTOMER CHURN DISTRIBUTION
# =========================================================

st.subheader(
    "👥 Customer Churn Distribution"
)


churn_distribution = (
    filtered_df["Churn"]
    .value_counts()
    .reset_index()
)


churn_distribution.columns = [
    "Churn",
    "Customers"
]


fig_pie = px.pie(
    churn_distribution,
    names="Churn",
    values="Customers",
    title="Customer Churn Distribution",
    hole=0.45
)


st.plotly_chart(
    fig_pie,
    width="stretch"
)


# =========================================================
# BUSINESS INSIGHT
# =========================================================

st.divider()

st.subheader(
    "💡 Business Insight"
)


if churn_rate >= 40:

    st.warning(
        f"⚠️ High churn risk detected. "
        f"Current filtered churn rate is "
        f"{churn_rate:.2f}%."
    )

elif churn_rate >= 25:

    st.info(
        f"ℹ️ Moderate churn level. "
        f"Current churn rate is "
        f"{churn_rate:.2f}%."
    )

else:

    st.success(
        f"✅ Relatively low churn level. "
        f"Current churn rate is "
        f"{churn_rate:.2f}%."
    )


# =========================================================
# FILTER SUMMARY
# =========================================================

st.divider()

st.subheader(
    "📋 Current Filter Summary"
)


st.write(
    f"Showing **{len(filtered_df):,} customers** "
    f"based on the selected filters."
)


# =========================================================
# DATA PREVIEW
# =========================================================

with st.expander(
    "🔎 View Customer Data"
):

    st.dataframe(
        filtered_df,
        width="stretch",
        height=400
    )