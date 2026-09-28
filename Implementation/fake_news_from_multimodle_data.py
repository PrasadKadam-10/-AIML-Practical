# Fake News Detection - Text + Metadata (Error-Free, No Downloads)

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------- Sample Data ----------------
texts = [
    "Breaking news: new policy announced",
    "Fake news: celebrity scandal",
    "Economy is growing steadily",
    "Fake: aliens discovered in desert",
    "Sports event draws large crowd",
    "Fake report about virus outbreak"
]

metadata = [
    [0.9, 0.2],  # source reliability, shares
    [0.1, 0.8],
    [0.8, 0.4],
    [0.2, 0.9],
    [0.85, 0.5],
    [0.3, 0.7]
]

labels = [1, 0, 1, 0, 1, 0]  # 1=Real, 0=Fake

# ---------------- Text Features ----------------
vectorizer = TfidfVectorizer(max_features=50)
text_features = vectorizer.fit_transform(texts).toarray()

# ---------------- Metadata Features ----------------
scaler = StandardScaler()
meta_features = scaler.fit_transform(metadata)

# ---------------- Combine Features ----------------
X = np.hstack((text_features, meta_features))
y = np.array(labels)

# ---------------- Train/Test Split ----------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------- Train Classifier ----------------
model = LogisticRegression()
model.fit(X_train, y_train)

# ---------------- Prediction & Evaluation ----------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ---------------- Predict New Example ----------------
new_text = ["Fake news: politician involved in scandal"]
new_meta = [[0.2, 0.7]]

new_text_features = vectorizer.transform(new_text).toarray()
new_meta_features = scaler.transform(new_meta)
new_X = np.hstack((new_text_features, new_meta_features))

pred = model.predict_proba(new_X)[:,1]  # probability of being Real
print("Predicted probability (Real=1, Fake=0):", pred[0])
