# -----------------------------------------------
# HYBRID MOVIE RECOMMENDATION SYSTEM
# Using Content-Based + Collaborative Filtering
# -----------------------------------------------

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

# ---------------------------
# Step 1: Load Dataset
# ---------------------------
movies = pd.read_csv("/content/RS-A2_A3_movie.csv")   # movieId, title, genres
tags = pd.read_csv("/content/RS-A2_A3_tag.csv")       # userId, movieId, tag, timestamp
ratings = pd.read_csv("/content/RS-A2_A3_Filtered_Ratings.csv")  # userId, movieId, rating, timestamp

# Clean and merge tags into movies
tags["tag"] = tags["tag"].astype(str)  # Convert all to string to avoid join errors
tags_grouped = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x)).reset_index()
merged_df = pd.merge(movies, tags_grouped, on="movieId", how="left")
merged_df["tag"] = merged_df["tag"].fillna("")

# ---------------------------
# Step 2: Content-Based Filtering
# ---------------------------
# Combine 'genres' and 'tag' to form content description
merged_df["content"] = merged_df["genres"].fillna("") + " " + merged_df["tag"]

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(merged_df["content"])

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# ---------------------------
# Step 3: Collaborative Filtering (SVD)
# ---------------------------
# Create user-item matrix
user_item_matrix = ratings.pivot(index="userId", columns="movieId", values="rating").fillna(0)
user_item_sparse = csr_matrix(user_item_matrix.values)

# Apply SVD for latent features
svd = TruncatedSVD(n_components=10, random_state=42)
latent_matrix = svd.fit_transform(user_item_sparse)

# ---------------------------
# Step 4: Define Recommendation Functions
# ---------------------------
def get_content_based_recommendations(title, cosine_sim=cosine_sim):
    """Returns top 5 movies similar to the given title based on content (genres + tags)."""
    if title not in merged_df["title"].values:
        print("❌ Movie not found in dataset.")
        return []
    idx = merged_df.index[merged_df["title"] == title][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:6]  # top 5 excluding itself
    movie_indices = [i[0] for i in sim_scores]
    return merged_df["title"].iloc[movie_indices].tolist()


def get_collaborative_recommendations(user_id):
    """Returns movies liked by users similar to the given user."""
    if user_id not in user_item_matrix.index:
        print("❌ User not found in dataset.")
        return []
    user_idx = list(user_item_matrix.index).index(user_id)
    similarities = pairwise_distances(latent_matrix[user_idx].reshape(1, -1), latent_matrix, metric="cosine")[0]
    similar_users = similarities.argsort()[1:6]  # top 5 similar users
    recommended_movies = []
    for idx in similar_users:
        user = user_item_matrix.index[idx]
        user_movies = ratings[ratings["userId"] == user]["movieId"].tolist()
        recommended_movies.extend(user_movies)
    recommended_titles = merged_df[merged_df["movieId"].isin(recommended_movies)]["title"].unique().tolist()
    return recommended_titles[:5]


def hybrid_recommendations(user_id, title):
    """Combine content-based and collaborative recommendations."""
    content_based = get_content_based_recommendations(title)
    collaborative_based = get_collaborative_recommendations(user_id)
    combined = list(set(content_based + collaborative_based))
    return combined[:5]

# ---------------------------
# Step 5: Get User Input
# ---------------------------
print("🎬 Welcome to the Hybrid Movie Recommendation System!")
user_id = int(input("Enter your User ID: "))
movie_title = input("Enter a movie title you like (e.g., Toy Story (1995)): ")

# ---------------------------
# Step 6: Generate and Display Recommendations
# ---------------------------
recommendations = hybrid_recommendations(user_id, movie_title)

if recommendations:
    print("\n---------------- Recommended Movies ----------------")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")
    print("---------------------------------------------------")
else:
    print("No recommendations found.")
