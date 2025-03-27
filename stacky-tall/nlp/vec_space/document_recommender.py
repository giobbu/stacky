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
documents_ = [
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


# apply knn to recommend documents
from sklearn.neighbors import NearestNeighbors

# Create a NearestNeighbors object
knn = NearestNeighbors(n_neighbors=3, algorithm='auto', metric='cosine')

# Fit NearestNeighbors to the TF-IDF matrix
knn.fit(tfidf_matrix)

# Predict the nearest neighbors for a new document
new_document = ["Machine learning is the disgusting future"]
new_tfidf = vectorizer.transform(new_document)

# Make a prediction
distances, indices = knn.kneighbors(new_tfidf)
# Display the recommended documents
recommended_docs = tfidf_df.index[indices.flatten()]
print("Recommended documents for the new document:")

for i in indices.flatten():
    print('------')
    print(documents[i])
    print('------')

