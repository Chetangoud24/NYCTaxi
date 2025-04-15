import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="NYC Taxi Trip Analysis",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function with caching
@st.cache_data
def load_data():
    df = pd.read_parquet('green_tripdata_2024-06.parquet')
    
    # Convert datetime columns
    df['lpep_pickup_datetime'] = pd.to_datetime(df['lpep_pickup_datetime'])
    df['lpep_dropoff_datetime'] = pd.to_datetime(df['lpep_dropoff_datetime'])
    
    # Calculate trip duration in minutes
    df['trip_duration'] = (df['lpep_dropoff_datetime'] - df['lpep_pickup_datetime']).dt.total_seconds() / 60
    
    # Extract weekday and hour
    df['weekday'] = df['lpep_dropoff_datetime'].dt.day_name()
    df['hourofday'] = df['lpep_dropoff_datetime'].dt.hour
    
    # Drop outliers in trip duration and distance
    df = df[(df['trip_duration'] > 0) & (df['trip_duration'] < 180)]
    df = df[(df['trip_distance'] > 0) & (df['trip_distance'] < 50)]
    
    return df

# Load the data
df = load_data()

# Sidebar for user inputs
st.sidebar.title("Filter Options")

# Date range filter
min_date = df['lpep_pickup_datetime'].min().date()
max_date = df['lpep_pickup_datetime'].max().date()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date],
    min_value=min_date,
    max_value=max_date
)

if len(date_range) == 2:
    start_date, end_date = date_range
    df = df[(df['lpep_pickup_datetime'].dt.date >= start_date) & 
            (df['lpep_pickup_datetime'].dt.date <= end_date)]

# ✅ Trip distance filter (FIXED missing parenthesis)
min_dist, max_dist = st.sidebar.slider(
    "Trip Distance (miles)",
    float(df['trip_distance'].min()),
    float(df['trip_distance'].max()),
    (float(df['trip_distance'].min()), float(df['trip_distance'].max()))
)
df = df[(df['trip_distance'] >= min_dist) & (df['trip_distance'] <= max_dist)]

# Trip duration filter
min_dur, max_dur = st.sidebar.slider(
    "Trip Duration (minutes)",
    int(df['trip_duration'].min()),
    int(df['trip_duration'].max()),
    (int(df['trip_duration'].min()), int(df['trip_duration'].max()))
)
df = df[(df['trip_duration'] >= min_dur) & (df['trip_duration'] <= max_dur)]

# Payment type filter
payment_options = ['All'] + list(df['payment_type'].dropna().unique())
selected_payment = st.sidebar.selectbox("Payment Type", payment_options)
if selected_payment != 'All':
    df = df[df['payment_type'] == selected_payment]

# Main content
st.title("🚖 NYC Taxi Trip Analysis")
st.markdown("Analyzing green taxi trip data from June 2024")

# Key metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Trips", f"{len(df):,}")
col2.metric("Avg. Distance", f"{df['trip_distance'].mean():.2f} miles")
col3.metric("Avg. Duration", f"{df['trip_duration'].mean():.2f} mins")
col4.metric("Avg. Fare", f"${df['total_amount'].mean():.2f}")

st.markdown("---")

# Tab layout
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Time Analysis", "Location Analysis", "Fare Analysis"])

# Tabs code remains the same
# ...

# Add some space at the bottom
st.markdown("---")
st.markdown("### Data Summary")
st.dataframe(df.describe(), use_container_width=True)

# Download button for filtered data
st.sidebar.markdown("---")
st.sidebar.download_button(
    label="Download Filtered Data as CSV",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name='filtered_nyc_taxi_data.csv',
    mime='text/csv'
)
