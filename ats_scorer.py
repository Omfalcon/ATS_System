import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from nltk.corpus import stopwords
import re

# NLTK data download
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')


class ATSScorer:
    def __init__(self):
        self.bert_model = SentenceTransformer('all-MiniLM-L6-v2')

        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            self.nlp = None

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )

        self.ats_keywords = self.load_keywords('data/ats_keywords.txt')
        self.technical_skills = self.load_keywords('data/skills_list.txt')
        self.stop_words = set(stopwords.words('english'))

    def load_keywords(self, filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                keywords = [line.strip().lower() for line in f if line.strip()]
            return set(keywords)
        except:
            return set()

    def calculate_ats_score(self, resume_text, jd_text):
        scores = {}

        # 1. TF-IDF Similarity
        scores['tfidf'] = self.tfidf_cosine_similarity(resume_text, jd_text)

        # 2. BERT Similarity
        scores['bert'] = self.bert_semantic_similarity(resume_text, jd_text)

        # 3. Keyword Match
        scores['keyword'] = self.keyword_match_percentage(resume_text, jd_text)

        # 4. Skills Overlap
        scores['skills'] = self.skills_overlap(resume_text, jd_text)

        # Final Score Calculation
        final_score = (
                              scores['tfidf'] * 0.3 +
                              scores['bert'] * 0.4 +
                              scores['keyword'] * 0.2 +
                              scores['skills'] * 0.1
                      ) * 100

        # Missing keywords
        missing_keywords = self.get_missing_keywords(resume_text, jd_text)

        return {
            'final_score': round(final_score, 2),
            'breakdown': {
                'TF-IDF Similarity': round(scores['tfidf'] * 100, 2),
                'BERT Semantic Similarity': round(scores['bert'] * 100, 2),
                'Keyword Match': round(scores['keyword'] * 100, 2),
                'Skills Overlap': round(scores['skills'] * 100, 2)
            },
            'missing_keywords': missing_keywords
        }

    def tfidf_cosine_similarity(self, resume, jd):
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform([resume, jd])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
            return float(similarity[0][0])
        except:
            return 0.0

    def bert_semantic_similarity(self, resume, jd):
        try:
            embeddings = self.bert_model.encode([resume, jd])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])
            return float(similarity[0][0])
        except:
            return 0.0

    def keyword_match_percentage(self, resume, jd):
        jd_keywords = self.extract_important_keywords(jd)
        if not jd_keywords:
            return 0.0

        matched = 0
        for keyword in jd_keywords:
            if keyword in resume.lower():
                matched += 1

        return matched / len(jd_keywords)

    def skills_overlap(self, resume, jd):
        resume_skills = [skill for skill in self.technical_skills if skill in resume.lower()]
        jd_skills = [skill for skill in self.technical_skills if skill in jd.lower()]

        if not jd_skills:
            return 0.5

        return len(resume_skills) / len(jd_skills)

    def extract_important_keywords(self, text):
        important_keywords = set()

        # ATS keywords
        for keyword in self.ats_keywords:
            if keyword in text.lower():
                important_keywords.add(keyword)

        # Technical skills
        for skill in self.technical_skills:
            if skill in text.lower():
                important_keywords.add(skill)

        # spaCy extraction
        if self.nlp and len(text) > 10:
            try:
                doc = self.nlp(text[:1000])
                for token in doc:
                    if (token.pos_ in ['NOUN', 'PROPN'] and
                            len(token.text) > 3 and
                            token.text.lower() not in self.stop_words):
                        important_keywords.add(token.text.lower())
            except:
                pass

        return list(important_keywords)[:20]

    def get_missing_keywords(self, resume_text, jd_text, top_n=8):
        jd_keywords = self.extract_important_keywords(jd_text)
        missing = []

        for keyword in jd_keywords:
            if keyword not in resume_text.lower():
                missing.append(keyword)

        return missing[:top_n]