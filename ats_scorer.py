import math
import re
import numpy as np
from collections import Counter
try:
    from ml.predict_knn import predict_ats_score
except:
    from .ml.predict_knn import predict_ats_score



class ATSScorer:
    def __init__(self):
        self.technical_skills = self.load_skills_from_file('data/skills_list.txt')

        # Minimal stop words only
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    def calculate_ats_score(self, resume_text, jd_text):
        """
        SIMPLIFIED VERSION - Cosine similarity ko fix karo
        """
        print("=== DEBUG START ===")

        # Step 1: Basic preprocessing (stop words mat hatao)
        resume_clean = self.simple_clean(resume_text)
        jd_clean = self.simple_clean(jd_text)

        # Step 2: Use simple word-based cosine similarity
        cosine_score = self.simple_cosine_similarity(resume_clean, jd_clean)
        print(f"Simple Cosine Similarity: {cosine_score}")

        # Step 3: Keyword Match Score
        keyword_score = self.keyword_match_score(resume_clean, jd_clean)
        print(f"Keyword Match: {keyword_score}")

        # Step 4: Skills Overlap
        skills_score = self.skills_overlap(resume_clean, jd_clean)
        print(f"Skills Overlap: {skills_score}")

        # Final Score - Adjusted weights
        final_score = (
                              cosine_score * 0.4 +  # Simple Cosine Similarity (40%)
                              keyword_score * 0.4 +  # Keyword Match (40%)
                              skills_score * 0.2  # Skills Overlap (20%)
                      ) * 100

        suggestions = self.get_suggestions(resume_clean, jd_clean)
        try:
            print("Attempting to predict using KNN model...")
            knn_score = predict_ats_score(resume_text, jd_text)
            print(f"KNN Model Score: {knn_score}")
            final_score = (final_score * 0.7) + (knn_score * 0.3)
            print("✅ KNN block executed successfully!")
        except Exception as e:
            print("⚠️ KNN model not found or failed:", e)

        print(f"Final Score: {final_score}")
        print("=== DEBUG END ===")

        return {
            'final_score': round(final_score, 2),
            'breakdown': {
                'Cosine Similarity': round(cosine_score * 100, 2),
                'Keyword Match': round(keyword_score * 100, 2),
                'Skills Overlap': round(skills_score * 100, 2)
            },
            'missing_keywords': suggestions
        }

    def simple_clean(self, text):
        """Minimal cleaning - technical words preserve karo"""
        if not text:
            return ""

        text = text.lower()
        # Remove special characters but keep technical symbols
        text = re.sub(r'[^\w\s\.\-\+]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def simple_cosine_similarity(self, text1, text2):
        """
        Simple word frequency based cosine similarity
        No TF-IDF - direct word counts use karo
        """
        if not text1 or not text2:
            return 0.0

        # Get words from both texts
        words1 = text1.split()
        words2 = text2.split()

        if not words1 or not words2:
            return 0.0

        # Create vocabulary from both texts
        vocabulary = set(words1 + words2)

        # Create frequency vectors
        freq1 = Counter(words1)
        freq2 = Counter(words2)

        # Create vectors
        vec1 = [freq1.get(word, 0) for word in vocabulary]
        vec2 = [freq2.get(word, 0) for word in vocabulary]

        # Convert to numpy arrays
        v1 = np.array(vec1)
        v2 = np.array(vec2)

        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)

        if mag1 == 0 or mag2 == 0:
            return 0.0

        similarity = dot_product / (mag1 * mag2)
        return max(0, min(1, similarity))

    def keyword_match_score(self, resume, jd):
        """Improved keyword matching"""
        # Extract meaningful words (3+ characters, not stop words)
        jd_words = [word for word in jd.split() if len(word) >= 3 and word not in self.stop_words]
        resume_words = set(resume.split())

        if not jd_words:
            return 0.0

        matched = sum(1 for word in jd_words if word in resume_words)
        return matched / len(jd_words)

    def extract_skills_from_text(self, text):
        """Skills extract with better matching"""
        found_skills = []
        for skill in self.technical_skills:
            # Use word boundaries for exact matching
            if re.search(r'\b' + re.escape(skill) + r'\b', text):
                found_skills.append(skill)
        return found_skills

    def skills_overlap(self, resume, jd):
        """Skills overlap with better handling"""
        jd_skills = self.extract_skills_from_text(jd)
        resume_skills = self.extract_skills_from_text(resume)

        print(f"JD Skills: {jd_skills}")
        print(f"Resume Skills: {resume_skills}")

        if not jd_skills:
            return 0.0

        common_skills = set(resume_skills) & set(jd_skills)
        return len(common_skills) / len(jd_skills)

    def get_suggestions(self, resume, jd):
        """Better suggestions"""
        jd_skills = self.extract_skills_from_text(jd)
        resume_skills = self.extract_skills_from_text(resume)

        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]

        # Also get important keywords from JD that are missing
        jd_words = [word for word in jd.split() if len(word) >= 4 and word not in self.stop_words]
        resume_words = set(resume.split())
        missing_keywords = [word for word in jd_words if word not in resume_words][:5]

        return missing_skills + missing_keywords[:8]

    def load_skills_from_file(self, filepath):
        """Load skills from file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                skills = [line.strip().lower() for line in f if line.strip()]
            return set(skills)
        except:
            return set()