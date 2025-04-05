import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(page_title="NYC Green Taxi Analysis", layout="wide")
st.title("🚕 NYC Green Taxi Data Analysis & Prediction")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Drop unused column
    df.drop(columns=['ehail_fee'], inplace=True, errors='ignore')

    # Parse datetime
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'], errors='coerce')
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'], errors='coerce')

    # Feature engineering
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
    df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour

    # Fill missing values
    num_cols = ['RatecodeID', 'passenger_count', 'payment_type', 'trip_type', 'congestion_surcharge']
    for col in num_cols:
        df[col].fillna(df[col].mean(), inplace=True)
    if 'store_and_fwd_flag' in df.columns:
        df['store_and_fwd_flag'].fillna(df['store_and_fwd_flag'].mode()[0], inplace=True)

    # Select features
    numeric_vars = [
        'trip_distance', 'fare_amount', 'extra', 'mta_tax', 'tip_amount',
        'tolls_amount', 'improvement_surcharge', 'congestion_surcharge',
        'trip_duration', 'passenger_count'
    ]
    object_vars = [
        'store_and_fwd_flag', 'RatecodeID', 'payment_type',
        'trip_type', 'weekday', 'hourofday'
    ]

    st.subheader("📊 Correlation Heatmap")
    corr_matrix = df[numeric_vars].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("💵 Distribution of Total Amount")
    fig, axs = plt.subplots(1, 3, figsize=(18, 4))
    df['total_amount'].hist(bins=50, ax=axs[0])
    axs[0].set_title("Histogram")
    sns.boxplot(x=df['total_amount'], ax=axs[1])
    axs[1].set_title("Boxplot")
    df['total_amount'].plot(kind='kde', ax=axs[2])
    axs[2].set_title("Density Plot")
    st.pyplot(fig)

    # Encode object vars
    df_encoded = pd.get_dummies(df, columns=object_vars, drop_first=True)

    # Prepare data
    X = df_encoded.drop(columns=['total_amount'], errors='ignore')
    y = df_encoded['total_amount']
    X = X.apply(pd.to_numeric, errors='coerce')
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.subheader("📈 Model Performance")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    st.write("**Linear Regression R²:**", r2_score(y_test, y_pred_lr))

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    y_pred_dt = dt.predict(X_test)
    st.write("**Decision Tree R²:**", r2_score(y_test, y_pred_dt))

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    st.write("**Random Forest R²:**", r2_score(y_test, y_pred_rf))

    # Gradient Boosting
    gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
    gb.fit(X_train, y_train)
    y_pred_gb = gb.predict(X_test)
    st.write("**Gradient Boosting R²:**", r2_score(y_test, y_pred_gb))

    st.success("✅ Analysis Complete")

else:
    st.warning("Please upload a CSV file to begin.")
