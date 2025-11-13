import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Step 1: Load dataset
data = pd.read_csv('../data/dataset.csv')

# Step 2: Combine resume + JD text for similarity learning
data['combined'] = data['resume_text'] + ' ' + data['jd_text']

# Step 3: Convert text → numbers using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
X = vectorizer.fit_transform(data['combined'])
y = data['label']

# Step 4: Train-test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Train KNN
model = KNeighborsClassifier(n_neighbors=3, metric='cosine')
model.fit(X_train, y_train)

# Step 6: Evaluate
pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)
print(f"✅ Model trained successfully! Accuracy: {acc*100:.2f}%")

# Step 7: Save model and vectorizer
joblib.dump(model, 'models/knn_model.pkl')
joblib.dump(vectorizer, 'models/tfidf.pkl')
print("📁 Model and vectorizer saved in models/ folder")
