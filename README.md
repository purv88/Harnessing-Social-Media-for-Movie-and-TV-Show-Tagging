Emotional Arc Analysis of IMDB Reviews


Overview
This project aims to analyze the emotional arc of movie reviews from IMDB using sentiment analysis and clustering techniques. The primary objectives are to visualize the sentiment trends over time and to identify distinct clusters of reviews based on their textual content.

Introduction
The Emotional Arc Analysis of IMDB Reviews project leverages natural language processing (NLP) and machine learning techniques to extract insights from user reviews. We perform sentiment analysis to calculate the sentiment polarity of each review and cluster the reviews to find patterns in the data.

Data Description
The dataset imdb_reviews.csv should contain at least the following columns:

review: The text content of the review.
date: The date the review was posted.

Installation
To run this project, ensure you have the following Python libraries installed:

pandas
plotly
textblob
scikit-learn

Usage
Load the dataset: Ensure your dataset (imdb_reviews.csv) is in the same directory as your script.

Preprocess the data: The script filters out reviews without text, converts review dates to datetime format, and sorts the data by date.

Sentiment Analysis: Sentiment polarity scores are calculated for each review using TextBlob.

Visualization: The script visualizes the sentiment scores over time using Plotly, with a trend line and a yearly average line.

Clustering: The reviews are vectorized using TF-IDF and clustered using k-means clustering to identify distinct groups based on text content.

Analysis
Sentiment Analysis
Sentiment Polarity Calculation: Each review's sentiment polarity is calculated using TextBlob. The polarity score ranges from -1 (most negative) to +1 (most positive).
Trend Analysis: A linear regression model is fitted to determine the sentiment trend over time.
Visualization: The sentiment scores, trend line, and yearly averages are plotted using Plotly to visualize the emotional arc of the reviews.
Clustering
TF-IDF Vectorization: The text data is transformed into TF-IDF features to represent the importance of words in each review.
K-means Clustering: Reviews are clustered into 5 clusters to identify groups of similar reviews. The top terms in each cluster are displayed to provide insights into the content of each cluster.

Results
The project outputs include:

A Plotly visualization showing sentiment polarity over time, with a regression trend line and yearly averages.
Top terms for each cluster, providing insights into common themes in each group of reviews.
Examples of reviews in each cluster to better understand the clustering results.

