# -------------------------------------------------------------
# Advanced E-commerce Product Recommendation System using SVD
# -------------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import random
import seaborn as sns

# -------------------------------------------------------------
# Step 1: Load and Prepare the Dataset
# -------------------------------------------------------------
df = pd.read_csv("RS-A5_amazon_products_sales_data_cleaned (1).csv")

print(f"✅ Loaded dataset with shape: {df.shape}")
print("Columns:", list(df.columns))

# Handle missing values
df = df.dropna(subset=["product_title", "product_rating"])
df["product_rating"] = df["product_rating"].astype(float)

# Create product IDs
df["product_id"] = range(1, len(df) + 1)

# -------------------------------------------------------------
# Step 2: Simulate User-Item Interactions
# -------------------------------------------------------------
num_users = 50  # Simulate 50 users
unique_products = df["product_id"].unique()
user_ids = list(range(1, num_users + 1))

interaction_data = []
for user in user_ids:
    sampled_products = np.random.choice(unique_products, size=40, replace=False)
    for product in sampled_products:
        base_rating = df.loc[df["product_id"] == product, "product_rating"].values[0]
        rating = np.clip(np.random.normal(base_rating, 0.5), 1, 5)
        interaction_data.append([user, product, round(rating, 1)])

interactions = pd.DataFrame(interaction_data, columns=["user_id", "product_id", "rating"])
print(f"✅ Simulated user-item interactions: {interactions.shape}")

# -------------------------------------------------------------
# Step 3: Create User-Item Rating Matrix
# -------------------------------------------------------------
rating_matrix = interactions.pivot(index="user_id", columns="product_id", values="rating").fillna(0)
print(f"User-Item Matrix shape: {rating_matrix.shape}")

# -------------------------------------------------------------
# Step 4: Model Evaluation (RMSE vs Components)
# -------------------------------------------------------------
rmse_values = []
components_range = [5, 10, 15, 20, 25]

for n in components_range:
    svd = TruncatedSVD(n_components=n, random_state=42)
    latent_matrix = svd.fit_transform(rating_matrix)
    reconstructed = np.dot(latent_matrix, svd.components_)
    rmse = np.sqrt(mean_squared_error(rating_matrix.values.flatten(), reconstructed.flatten()))
    rmse_values.append(rmse)

plt.figure(figsize=(8, 5))
plt.plot(components_range, rmse_values, marker='o', linestyle='--')
plt.title("RMSE vs Number of Latent Features (Components)")
plt.xlabel("Number of Components")
plt.ylabel("RMSE")
plt.grid(True)
plt.show()

# -------------------------------------------------------------
# Step 5: Apply Optimal Matrix Factorization (SVD)
# -------------------------------------------------------------
best_components = components_range[np.argmin(rmse_values)]
print(f"🎯 Optimal number of components selected: {best_components}")

svd = TruncatedSVD(n_components=best_components, random_state=42)
latent_matrix = svd.fit_transform(rating_matrix)
reconstructed_matrix = np.dot(latent_matrix, svd.components_)

# Evaluate final model
rmse = np.sqrt(mean_squared_error(rating_matrix.values.flatten(), reconstructed_matrix.flatten()))
mae = mean_absolute_error(rating_matrix.values.flatten(), reconstructed_matrix.flatten())

print("\n📊 Final Model Evaluation:")
print(f"   RMSE: {rmse:.4f}")
print(f"   MAE : {mae:.4f}")

# -------------------------------------------------------------
# Step 6: Recommendation Function
# -------------------------------------------------------------
def recommend_products(user_id, num_recommendations=5):
    """Recommend top N products for a given user."""
    user_ratings = reconstructed_matrix[user_id - 1]
    rated_products = rating_matrix.loc[user_id][rating_matrix.loc[user_id] > 0].index
    recommendations = [
        (df.loc[df["product_id"] == pid, "product_title"].values[0], score)
        for pid, score in enumerate(user_ratings, start=1)
        if pid not in rated_products
    ]
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:num_recommendations]
    return recommendations

# -------------------------------------------------------------
# Step 7: Display Recommendations for a Sample User
# -------------------------------------------------------------
sample_user = random.choice(user_ids)
print(f"\n🛍️  Generating product recommendations for User ID: {sample_user}")

recommendations = recommend_products(sample_user, num_recommendations=5)
print("\n---------------- Recommended Products ----------------")
for i, (title, score) in enumerate(recommendations, 1):
    print(f"{i}. {title}  (Predicted Preference Score: {score:.2f})")
print("-------------------------------------------------------")

# -------------------------------------------------------------
# Step 8: Visualization – Top 5 Recommendations
# -------------------------------------------------------------
titles = [rec[0][:40] + "..." for rec in recommendations]
scores = [rec[1] for rec in recommendations]

plt.figure(figsize=(8, 5))
sns.barplot(x=scores, y=titles, palette="coolwarm")
plt.title(f"Top 5 Recommended Products for User {sample_user}")
plt.xlabel("Predicted Preference Score")
plt.ylabel("Product Title")
plt.show()

# -------------------------------------------------------------
# Step 9: Observations and Conclusions
# -------------------------------------------------------------
print("\n🧠 Observations and Conclusions:")
print("- The system successfully applies Matrix Factorization (SVD) to identify hidden relationships between users and products.")
print("- RMSE vs Components analysis helps select the optimal latent dimension for accurate predictions.")
print("- The recommendation bar chart visually demonstrates personalized product suggestions.")
print("- RMSE and MAE values indicate effective performance (lower = better).")
print("- The model can be further improved with actual user behavior data such as clicks, purchases, or ratings.")
print("- Integrating it into an e-commerce platform can enhance user experience and drive sales growth.")
