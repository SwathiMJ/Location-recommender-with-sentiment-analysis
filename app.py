import streamlit as st
import pandas as pd
from textblob import TextBlob
import plotly.express as px
from sklearn.cluster import KMeans
import io
import chardet
import numpy as np
import geopandas as gpd

# Function for sentiment evaluation
def sentiment_evaluation(review):
    # Handling missing or null values
    if pd.isnull(review) or review == '':
        return 'Neutral'
    # Analyzing the sentiment of the review
    sentiment = TextBlob(str(review)).sentiment
    # Classifying based on polarity
    if sentiment.polarity > 0.1:
        return 'Positive'
    elif sentiment.polarity < -0.1:
        return 'Negative'
    else:
        return 'Neutral'


def recommend_location(df, location1, location2):
    # Filter the DataFrame based on the provided store locations
    df_filtered = df[(df['store_location'] == location1) | (df['store_location'] == location2)]

    # Check if there are enough samples for clustering
    if len(df_filtered) < 5:
        return None, None

    # Extract latitude and longitude columns
    locations = df_filtered[['latitude', 'longitude']]

    # Perform clustering
    kmeans = KMeans(n_clusters=min(5, len(df_filtered)))  # Adjust number of clusters
    df_filtered['Cluster'] = kmeans.fit_predict(locations)

    # Find centroid of each cluster
    cluster_centers = kmeans.cluster_centers_

    # Find the top 3 clusters
    top_clusters = df_filtered['Cluster'].value_counts().nlargest(3).index

    # Initialize an empty list to store centroids of top clusters
    top_cluster_centroids = []

    # Find nearest location to the centroid of each top cluster
    for cluster in top_clusters:
        nearest_location_idx = locations[df_filtered['Cluster'] == cluster].apply(
            lambda row: ((row - cluster_centers[cluster]) ** 2).sum(),
            axis=1
        ).idxmin()
        top_cluster_centroids.append(cluster_centers[cluster])

    # Calculate the midpoint between the two input locations
    midpoint_latitude = np.mean([centroid[0] for centroid in top_cluster_centroids])
    midpoint_longitude = np.mean([centroid[1] for centroid in top_cluster_centroids])

    # Find the state name corresponding to the recommended location
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    point = gpd.points_from_xy([midpoint_longitude], [midpoint_latitude])
    nearest_country = world[world.geometry.distance(point[0]) == world.geometry.distance(point[0]).min()]
    state_name = nearest_country['name'].values[0]

    return (midpoint_latitude, midpoint_longitude, state_name), (location1, location2)


# Main function
def main():
    # Add custom CSS for background color
    st.markdown(
        """
        <style>
            body {
                background-color: #f5c6cb; /* Pale red */
            }
            .css-1aumxhk.ehghp {
                background-color: #f5c6cb; /* Pale red */
            }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title('Sentiment Analysis and Map Visualization')

    # Sidebar
    option = st.sidebar.radio("Choose an option:",
                              ("Classify Review", "Upload File", "Recommend Location"))

    if option == "Classify Review":
        st.header("Classify Review")
        review = st.text_area("Enter your review here:")
        if st.button("Classify"):
            sentiment = sentiment_evaluation(review)
            st.write("The sentiment of the review is:", sentiment)
    elif option == "Upload File":
        st.header("Upload File")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file is not None:
            try:
                # Detect encoding
                raw_data = uploaded_file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

                # Decode with detected encoding
                data = raw_data.decode(encoding)
                df = pd.read_csv(io.StringIO(data))
                st.session_state.df = df
                st.success("File uploaded successfully.")

                # Plot the graph after uploading the file
                st.header("Plot Graph")
                st.write("Plot graph section is executing...")  # Debug statement
                # Ensure df is defined
                if 'df' not in st.session_state:
                    st.warning("Please upload a CSV file first.")
                else:
                    df = st.session_state.df
                    # Apply sentiment evaluation to create the "Sentiments" column
                    df['Sentiments'] = df['review'].apply(sentiment_evaluation)

                    # Plot the input location with sentiment analysis
                    fig_input = px.scatter_mapbox(df,
                                                  lat="latitude",
                                                  lon="longitude",
                                                  hover_name="Sentiments",
                                                  color="store_location",
                                                  color_continuous_scale=px.colors.sequential.RdBu,
                                                  size_max=5,
                                                  zoom=10)
                    fig_input.update_layout(mapbox_style="carto-positron")
                    fig_input.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                    st.subheader("Input Location with Sentiment Analysis")
                    st.plotly_chart(fig_input)

            except Exception as e:
                st.error(f"Error: {e}")

    elif option == "Recommend Location":
        st.header("Recommend Location")
        # Ensure df is defined
        if 'df' not in st.session_state:
            st.warning("Please upload a CSV file first.")
        else:
            location1 = st.text_input("Enter the first store location name:")
            location2 = st.text_input("Enter the second store location name:")
            if st.button("Recommend"):
                st.write("Recommendation section is executing...")  # Debug statement
                recommended_location, input_locations = recommend_location(st.session_state.df, location1, location2)
                print("Recommended Location:", recommended_location)  # Debug statement
                if recommended_location is not None:
                    # Plot the input locations and the recommended location on the map
                    input_locations_df = st.session_state.df[
                        (st.session_state.df['store_location'] == location1) |
                        (st.session_state.df['store_location'] == location2)
                    ]
                    input_locations_df['Sentiments'] = input_locations_df['store_location']
                    input_locations_df['latitude'] = input_locations_df['latitude'].astype(float)
                    input_locations_df['longitude'] = input_locations_df['longitude'].astype(float)
                    input_locations_df = input_locations_df[['latitude', 'longitude', 'Sentiments']]
                    input_locations_df = pd.concat([input_locations_df, pd.DataFrame({
                        'latitude': [recommended_location[0]],
                        'longitude': [recommended_location[1]],
                        'Sentiments': ['Recommended Location: ' + recommended_location[2]]
                    })])

                    fig_recommendation = px.scatter_mapbox(input_locations_df,
                                                           lat='latitude',
                                                           lon='longitude',
                                                           hover_name='Sentiments',
                                                           zoom=10)
                    fig_recommendation.update_layout(mapbox_style="carto-positron")
                    fig_recommendation.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
                    st.subheader("Input and Recommended Locations")
                    st.plotly_chart(fig_recommendation)
                else:
                    st.warning("No data available for the provided store locations.")


if __name__ == "__main__":
    main()