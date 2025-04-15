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
    df = pd.read_csv('green_tripdata_2024-06.csv')
    
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

# Trip distance filter
min_dist, max_dist = st.sidebar.slider(
    "Trip Distance (miles)",
    float(df['trip_distance'].min()),
    float(df['trip_distance'].max()),
    (float(df['trip_distance'].min()), float(df['trip_distance'].max()))
df = df[(df['trip_distance'] >= min_dist) & (df['trip_distance'] <= max_dist)]

# Trip duration filter
min_dur, max_dur = st.sidebar.slider(
    "Trip Duration (minutes)",
    int(df['trip_duration'].min()),
    int(df['trip_duration'].max()),
    (int(df['trip_duration'].min()), int(df['trip_duration'].max())))
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

with tab1:
    st.header("Trip Overview")
    
    # Distribution plots
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.subheader("Trip Distance Distribution")
        fig = px.histogram(df, x='trip_distance', nbins=50, 
                          labels={'trip_distance': 'Trip Distance (miles)'},
                          color_discrete_sequence=['#00CC96'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    with fig_col2:
        st.subheader("Trip Duration Distribution")
        fig = px.histogram(df, x='trip_duration', nbins=50,
                          labels={'trip_duration': 'Trip Duration (minutes)'},
                          color_discrete_sequence=['#EF553B'])
        fig.update_layout(bargap=0.1)
        st.plotly_chart(fig, use_container_width=True)
    
    # Payment type and passenger count
    fig_col3, fig_col4 = st.columns(2)
    
    with fig_col3:
        st.subheader("Payment Type Distribution")
        payment_counts = df['payment_type'].value_counts().reset_index()
        payment_counts.columns = ['Payment Type', 'Count']
        fig = px.pie(payment_counts, values='Count', names='Payment Type',
                    color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)
    
    with fig_col4:
        st.subheader("Passenger Count Distribution")
        passenger_counts = df['passenger_count'].value_counts().reset_index()
        passenger_counts.columns = ['Passenger Count', 'Count']
        fig = px.bar(passenger_counts, x='Passenger Count', y='Count',
                    labels={'Count': 'Number of Trips'},
                    color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.header("Time-Based Analysis")
    
    # Weekday analysis
    st.subheader("Trips by Weekday")
    weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekday_counts = df['weekday'].value_counts().reindex(weekday_order).reset_index()
    weekday_counts.columns = ['Weekday', 'Count']
    
    fig = px.bar(weekday_counts, x='Weekday', y='Count', 
                labels={'Count': 'Number of Trips'},
                color='Weekday',
                color_discrete_sequence=px.colors.qualitative.Pastel)
    st.plotly_chart(fig, use_container_width=True)
    
    # Hourly analysis
    st.subheader("Trips by Hour of Day")
    hour_counts = df['hourofday'].value_counts().sort_index().reset_index()
    hour_counts.columns = ['Hour', 'Count']
    
    fig = px.line(hour_counts, x='Hour', y='Count', 
                 labels={'Count': 'Number of Trips'},
                 markers=True)
    fig.update_traces(line_color='#AB63FA', line_width=2.5)
    st.plotly_chart(fig, use_container_width=True)
    
    # Time vs. metrics
    st.subheader("Time vs. Trip Metrics")
    time_metric = st.selectbox("Select Metric", ['trip_distance', 'trip_duration', 'total_amount'])
    
    fig = px.box(df, x='hourofday', y=time_metric,
                labels={
                    'hourofday': 'Hour of Day',
                    'trip_distance': 'Trip Distance (miles)',
                    'trip_duration': 'Trip Duration (minutes)',
                    'total_amount': 'Total Amount ($)'
                })
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.header("Location Analysis")
    
    # Top pickup and dropoff locations
    top_n = st.slider("Select number of top locations to display", 5, 20, 10)
    
    fig_col1, fig_col2 = st.columns(2)
    
    with fig_col1:
        st.subheader(f"Top {top_n} Pickup Locations")
        top_pickups = df['PULocationID'].value_counts().head(top_n).reset_index()
        top_pickups.columns = ['Location ID', 'Count']
        fig = px.bar(top_pickups, x='Location ID', y='Count',
                    labels={'Count': 'Number of Pickups'},
                    color='Count',
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
    
    with fig_col2:
        st.subheader(f"Top {top_n} Dropoff Locations")
        top_dropoffs = df['DOLocationID'].value_counts().head(top_n).reset_index()
        top_dropoffs.columns = ['Location ID', 'Count']
        fig = px.bar(top_dropoffs, x='Location ID', y='Count',
                    labels={'Count': 'Number of Dropoffs'},
                    color='Count',
                    color_continuous_scale='Plasma')
        st.plotly_chart(fig, use_container_width=True)
    
    # Location vs. metrics
    st.subheader("Location vs. Trip Metrics")
    loc_metric = st.selectbox("Select Metric", ['trip_distance', 'trip_duration', 'total_amount'], key='loc_metric')
    loc_type = st.radio("Location Type", ['Pickup', 'Dropoff'])
    
    if loc_type == 'Pickup':
        loc_col = 'PULocationID'
    else:
        loc_col = 'DOLocationID'
    
    # Get top 20 locations for better visualization
    top_locs = df[loc_col].value_counts().head(20).index
    filtered_df = df[df[loc_col].isin(top_locs)]
    
    fig = px.box(filtered_df, x=loc_col, y=loc_metric,
                labels={
                    loc_col: f'{loc_type} Location ID',
                    'trip_distance': 'Trip Distance (miles)',
                    'trip_duration': 'Trip Duration (minutes)',
                    'total_amount': 'Total Amount ($)'
                })
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Fare Analysis")
    
    # Fare components breakdown
    st.subheader("Fare Components Breakdown")
    fare_components = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge']
    avg_fare_components = df[fare_components].mean().reset_index()
    avg_fare_components.columns = ['Component', 'Average Amount']
    
    fig = px.bar(avg_fare_components, x='Component', y='Average Amount',
                labels={'Average Amount': 'Average Amount ($)'},
                color='Component')
    st.plotly_chart(fig, use_container_width=True)
    
    # Fare vs. distance/duration
    st.subheader("Fare Relationships")
    fare_rel = st.selectbox("Select Relationship", ['Fare vs. Distance', 'Fare vs. Duration'])
    
    if fare_rel == 'Fare vs. Distance':
        x_col = 'trip_distance'
        x_label = 'Trip Distance (miles)'
    else:
        x_col = 'trip_duration'
        x_label = 'Trip Duration (minutes)'
    
    # Sample the data for better performance in scatter plot
    sample_df = df.sample(min(1000, len(df)))
    
    fig = px.scatter(sample_df, x=x_col, y='total_amount',
                    labels={
                        x_col: x_label,
                        'total_amount': 'Total Fare ($)'
                    },
                    trendline="lowess",
                    color_discrete_sequence=['#00CC96'])
    st.plotly_chart(fig, use_container_width=True)
    
    # Payment type vs. tipping
    st.subheader("Tipping Behavior by Payment Type")
    fig = px.box(df, x='payment_type', y='tip_amount',
                labels={
                    'payment_type': 'Payment Type',
                    'tip_amount': 'Tip Amount ($)'
                })
    st.plotly_chart(fig, use_container_width=True)

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
