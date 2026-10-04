# Practical: Fake News Detection using Multimodal Data
# Objective: Classify news articles as Real or Fake by combining
#            TF-IDF text features with numeric metadata features.

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# --------------------------------------------------
# Step 1: Sample news articles and metadata
# --------------------------------------------------
articles = [
    "Government announces new education reform policy",
    "Shocking: celebrity arrested for fraud charges",
    "Stock market reaches record high this quarter",
    "Scientists find cure for common cold — exclusive",
    "National team wins championship title",
    "Hidden virus spreading in major cities undetected"
]

# Each row: [source_credibility (0-1), viral_share_rate (0-1)]
meta_info = [
    [0.88, 0.25],
    [0.15, 0.82],
    [0.82, 0.38],
    [0.18, 0.91],
    [0.80, 0.48],
    [0.25, 0.75]
]

# Labels: 1 = Real, 0 = Fake
news_labels = [1, 0, 1, 0, 1, 0]

# --------------------------------------------------
# Step 2: Extract TF-IDF features from article text
# --------------------------------------------------
tfidf = TfidfVectorizer(max_features=50)
text_feat = tfidf.fit_transform(articles).toarray()

# --------------------------------------------------
# Step 3: Scale the metadata features
# --------------------------------------------------
scaler = StandardScaler()
meta_feat = scaler.fit_transform(meta_info)

# --------------------------------------------------
# Step 4: Combine text + metadata into one feature matrix
# --------------------------------------------------
X = np.hstack((text_feat, meta_feat))
y = np.array(news_labels)

# --------------------------------------------------
# Step 5: Train / test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7)

# --------------------------------------------------
# Step 6: Train Logistic Regression classifier
# --------------------------------------------------
clf = LogisticRegression(max_iter=200)
clf.fit(X_train, y_train)

# --------------------------------------------------
# Step 7: Evaluate on test set
# --------------------------------------------------
predictions = clf.predict(X_test)
print("Accuracy :", accuracy_score(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))

# --------------------------------------------------
# Step 8: Predict on a new unseen article
# --------------------------------------------------
sample_article = ["Politician caught in corruption scandal — exclusive report"]
sample_meta    = [[0.18, 0.78]]

s_text = tfidf.transform(sample_article).toarray()
s_meta = scaler.transform(sample_meta)
s_X    = np.hstack((s_text, s_meta))

prob_real = clf.predict_proba(s_X)[0][1]
print(f"\nNew article — Probability of being Real: {prob_real:.4f}")
print("Prediction:", "Real" if prob_real >= 0.5 else "Fake")
