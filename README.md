
# Location Recommender with Sentiment Analysis

This project aims to leverage sentiment analysis and clustering techniques to extract meaningful insights from customer reviews. By doing so, it helps organizations improve revenues and optimize new branch locations. Advanced Natural Language Processing (NLP) techniques and deep learning models are employed to analyze sentiment, while clustering algorithms recommend optimal locations based on geographic data.

## Demo

![ezgif com-crop (1)](https://github.com/SwathiMJ/Location-recommender-with-sentiment-analysis/assets/140050536/0586941a-726d-4b4e-ab17-f0ae454fb418)


## Documentation

### Introduction

The Sentiment Analysis and Location Recommendation System is designed to analyze customer reviews and provide actionable insights. It classifies the sentiment expressed in the reviews and uses clustering techniques to recommend optimal locations for new branches based on geographic data.

### Features

* Sentiment analysis using advanced NLP techniques.
* Location recommendations based on geographic clustering.
* Interactive visualization of sentiment data and geographic distributions.
* Deployed on Streamlit Community Cloud for easy access and interaction.

### Installation

1. Clone the Repository: 

```bash
  git clone [https://github.com/SwathiMJ/sentiment-location-recommendation.git](https://github.com/SwathiMJ/Location-recommender-with-sentiment-analysis)

```
2. Install Dependencies:

```bash
  pip install -r requirements.txt
```
### Usage 

1. Run Preprocessing Script:

```bash
  python preprocess.py

```
2. Train the Models:
```bash
  python train_models.py
```
3. Perform Clustering:
```bash
  python cluster_reviews.py
```
4. Generate Recommendations:
```bash
  python recommend_locations.py

```
5. Visualize Results:
```bash
  python visualize_map.py
```

### Models and Techniques
#### Deep Learning Models
1. Simple Neural Network (SNN)

* Used for initial sentiment classification tasks.
2. Convolutional Neural Network (CNN)

* Effective for spatial data and text classification by capturing local patterns.
3. Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN)

* Handles sequential data, capturing long-term dependencies for better sentiment analysis.

#### Clustering Algorithm
* K-means Clustering Algorithm
* Groups reviews based on geographic data, assisting in location recommendations.

### Evaluation Metrics
1. Accuracy

* Ratio of correctly predicted instances to the total number of instances.
2. Precision

* Ratio of correctly predicted positive observations to the total predicted positives.
3. Recall (Sensitivity or True Positive Rate)

* Ratio of correctly predicted positive observations to all actual positives.
4. F1 Score

* Harmonic mean of precision and recall, balancing the two metrics.
5. Silhouette Score (for Clustering)

* Measures how similar an object is to its own cluster compared to other clusters.

### Advantages
1. Data-Driven Insights: 
Provides actionable insights by analyzing customer reviews and recommending optimal branch locations.

2. Improved Decision-Making:
 Enhances organizational decision-making with evidence-based recommendations.

3. Enhanced Customer Understanding:
 Helps organizations understand customer sentiment and preferences better.

4. Visual Representation: 
Integrates Mapbox to provide clear and intuitive visualizations of sentiment information and geographic distributions.

5. Ease of Use: 
Deployed on Streamlit Community Cloud, making it easily accessible and interactive for users without requiring extensive technical knowledge.

6. Advanced Modeling: 
Utilizes cutting-edge deep learning and clustering techniques to deliver accurate and reliable results.

### Disadvantages
1. Data Dependency: 
Relies heavily on the quality and quantity of customer review data.

2. Complexity: 
Requires technical expertise to set up and maintain.

3. Computational Resources: 
Resource-intensive, especially for large datasets.
## Appendix

### A. Data Description

The dataset used in this project includes the following columns:

1. Review Text:

* Definition: The textual content of the customer review.
* Use: Provides the raw data for sentiment analysis.
2. Rating:

* Definition: The rating given by the customer, typically on a scale (e.g., 1 to 5).
* Use: Acts as an additional indicator of customer sentiment.
3. Timestamp:

* Definition: The date and time when the review was posted.
* Use: Helps in analyzing trends over time.
4. Latitude:

* Definition: The geographic latitude of the review location.
* Use: Used for clustering reviews based on geographic location.
5. Longitude:

* Definition: The geographic longitude of the review location.
* Use: Used for clustering reviews based on geographic location.

### B. Model Descriptions
1. Simple Neural Network (SNN)

* A basic neural network model used for initial sentiment classification tasks.
2. Convolutional Neural Network (CNN)

* A deep learning model particularly effective for spatial data and text classification by capturing local patterns.
3. Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN)

* An advanced neural network model designed to handle sequential data, capturing long-term dependencies for better sentiment analysis.
4. K-means Clustering Algorithm

* A clustering technique used to group reviews based on geographic data, assisting in location recommendations.

### C. Tools and Technologies
* Pandas: Used for data manipulation and analysis.
* NumPy: Used for numerical computations and handling arrays.
* Scikit-learn: Used for machine learning algorithms, including K-means clustering.
* TensorFlow/Keras: Used for building and training deep learning models (SNN, CNN, LSTM).
* NLTK (Natural Language Toolkit): Used for natural language processing tasks.
* Mapbox: Used for visualizing geographic data and sentiment distribution.
* Streamlit: Used for deploying the application on Streamlit Community Cloud and creating interactive web interfaces.



## Deployment

The project is deployed on Streamlit Community Cloud. You can access and interact with the system using the following link:

https://swathimj-location-recommender-with-sentiment-analysi-app-kurwbi.streamlit.app/


## Authors

- @Swathimohan
- https://github.com/SwathiMJ/Location-recommender-with-sentiment-analysis

