import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans


# Load data
@st.cache
def load_data():
    return pd.read_csv('/TeePublic_review.csv')


df = load_data()

# Perform clustering
kmeans = KMeans(n_clusters=5)  # Adjust number of clusters as needed
df['Cluster'] = kmeans.fit_predict(df[['latitude', 'longitude']])

# Plot clusters
fig = px.scatter_mapbox(df, lat="latitude", lon="longitude", color="Cluster",
                        color_continuous_scale=px.colors.sequential.Viridis,
                        size_max=15, zoom=10)
fig.update_layout(mapbox_style="carto-positron")
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig)