import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Example corpus 1
documents = [
            "Machine learning is powerful",
            "Deep learning is a subset of machine learning",
            "Natural language processing is part of AI",
            "AI is evolving with deep learning",
            "Machine learning and AI are transforming industries",
            "Machine learning is the future",
            "Deep learning is the future",
            "AI is the future of nothing",
            "The dog is brown",
            "The cat is black",
            " Cat and AI are not the same",
            "The cat is a pet",
            "The dog is a pet",
        ]

# Example corpus 2
documents = [
            "Machine learning is powerful as my mother",
            "Deep learning is a subset of machine learning",
            "Natural language processing is part of AI",
            "AI is evolving with deep learning",
            "Machine learning and AI are transforming industries",
            "Machine learning is the future, they say",
            "Deep learning is the future, and it is not bright",
            "AI is the future of nothing",
            "The dog is brown as shit",
            "The cat is black",
            " Cat and AI are not the same",
            "The cat is a pet",
            "The dog is a pet with passion",
        ]

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Compute TF-IDF matrix
tfidf_matrix = vectorizer.fit_transform(documents)

# Convert to DataFrame for readability
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=[f"Doc{i+1}" for i in range(len(documents))], columns=vectorizer.get_feature_names_out())

# Display TF-IDF scores
print(tfidf_df)

# Create a PCA object
pca = PCA(n_components=2)

# Fit and transform the TF-IDF matrix
tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())

# Convert to DataFrame for readability
tfidf_pca_df = pd.DataFrame(tfidf_pca, index=[f"Doc{i+1}" for i in range(len(documents))], columns=["PC1", "PC2"])

plt.figure(figsize=(10, 6))
plt.scatter(tfidf_pca_df["PC1"], tfidf_pca_df["PC2"], color='blue')
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA of TF-IDF Matrix")
plt.grid(True)
for i, txt in enumerate(tfidf_pca_df.index):
    plt.annotate(txt, (tfidf_pca_df["PC1"][i], tfidf_pca_df["PC2"][i]))
plt.show()

# Create a KMeans object
kmeans = KMeans(n_clusters=3, random_state=42)

# Fit KMeans to the TF-IDF matrix
kmeans.fit(tfidf_pca_df)

# Add cluster labels to the DataFrame
tfidf_pca_df["Cluster"] = kmeans.labels_

# Plot the clustered documents
plt.figure(figsize=(10, 6))
colors = ['red', 'green', 'blue']
for cluster in range(3):
    cluster_df = tfidf_pca_df[tfidf_pca_df["Cluster"] == cluster]
    plt.scatter(cluster_df["PC1"], cluster_df["PC2"], color=colors[cluster], label=f"Cluster {cluster}")
    for i, txt in enumerate(cluster_df.index):
        plt.annotate(txt, (cluster_df["PC1"][i], cluster_df["PC2"][i]))
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("KMeans Clustering of Documents")
plt.legend()
plt.grid(True)
plt.show()