# Import needed libraries for processes
import pandas as pd
import plotly.graph_objects as go
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Load dataset reviews from CSV
df_reviews = pd.read_csv('imdb_reviews.csv')

# Filter out reviews without text
df_reviews = df_reviews[df_reviews['review'] != 'No review text']

# Convert review dates to datetime format
df_reviews['date'] = pd.to_datetime(df_reviews['date'], errors='coerce')
df_reviews = df_reviews.dropna(subset=['date'])

# Date sort the review timeline
df_reviews.sort_values('date', inplace=True)

# Calculate sentiment polarity scores for each review
df_reviews['polarity'] = df_reviews['review'].apply(lambda x: TextBlob(x).sentiment.polarity)

# Calculate the average sentiment polarity per year
df_reviews['year'] = df_reviews['date'].dt.year
yearly_avg = df_reviews.groupby('year')['polarity'].mean().reset_index()

# Linear Regression to find trend over the years
df_reviews['date_num'] = (df_reviews['date'] - df_reviews['date'].min()).dt.days
X = df_reviews[['date_num']]
y = df_reviews['polarity']
model = LinearRegression()
model.fit(X, y)
df_reviews['trend_polarity'] = model.predict(X)

# Plot polarity over time and visualize regression and moving average line using Plotly
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_reviews['date'], y=df_reviews['polarity'], mode='markers', name='Sentiment Scores'))
fig.add_trace(go.Scatter(x=df_reviews['date'], y=df_reviews['trend_polarity'], name='Trend Line'))
fig.add_trace(go.Scatter(x=pd.to_datetime(yearly_avg['year'], format='%Y'), y=yearly_avg['polarity'], mode='lines', name='Yearly Average', line=dict(color='green', width=2, dash='dash')))
fig.update_layout(title='Emotional Arc of Reviews Over Time',
                  xaxis_title='Date',
                  yaxis_title='Sentiment Polarity',
                  legend_title='Legend')
fig.show()

# Clustering using TF-IDF and k-means
text_data = df_reviews['review'].values
tfidf_vectorize = TfidfVectorizer(stop_words='english', max_features=1000)
tfidf_matrix = tfidf_vectorize.fit_transform(text_data)

# Define the number of clusters
num_clusters = 5
km = KMeans(n_clusters=num_clusters, random_state=42)
km.fit(tfidf_matrix)

# Assign cluster label to each review
df_reviews['cluster'] = km.labels_

# Display the top terms per cluster
order_centroids = km.cluster_centers_.argsort()[:, ::-1]
terms = tfidf_vectorize.get_feature_names_out()
print("Top terms per cluster:")
for i in range(num_clusters):
    top_ten_words = [terms[ind] for ind in order_centroids[i, :10]]
    print(f"Cluster {i}: {top_ten_words}")

# Review the clustering by examining the reviews per cluster
for i in range(num_clusters):
    print(f"\nCluster {i} reviews:")
    print(df_reviews[df_reviews['cluster'] == i]['review'].head())
