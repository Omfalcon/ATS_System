import joblib

# Load model & vectorizer once
model = joblib.load('models/knn_model.pkl')
vectorizer = joblib.load('models/tfidf.pkl')

def predict_ats_score(resume_text, jd_text):
    """Predict ATS score using trained KNN model"""
    combined = resume_text + ' ' + jd_text
    X = vectorizer.transform([combined])
    prob = model.predict_proba(X)[0][1] * 100  # Probability of 'match'
    return round(prob, 2)
