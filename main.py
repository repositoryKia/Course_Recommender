import pandas as pd
import string
import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Load dataset
df = pd.read_csv(r'D:\Bismillah SKRIPSIIIII\REVISI\dataset\dataset.csv')

# Clean text function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    if isinstance(text, str):
        text = re.sub(r'_+', ' ', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = text.lower()
        text = ''.join([c for c in text if c not in string.punctuation])
        text = re.sub(r'\d+', '', text)
        text = re.sub(' +', ' ', text).strip()
        text = ' '.join([lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words])
        return text
    return ""


tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Sequence'])

# Recommendation function
def recommend_courses(user_input, tfidf_matrix, df, top_n=10):
    user_input_processed = clean_text(user_input) 
    user_tfidf = tfidf_vectorizer.transform([user_input_processed])
    cosine_similarities = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    
    
    recommended_indices = cosine_similarities.argsort()[::-1]  
    recommendations = df.iloc[recommended_indices][['course_id', 'name', 'course_url', 'rating']]
    recommendations['similarity_score'] = cosine_similarities[recommended_indices]
    
    unique_recommendations = recommendations.drop_duplicates(subset='course_id').head(top_n)
    return unique_recommendations

def sidebar():
    with st.sidebar:
        st.title("💻Course Recommender")
        st.markdown("Find the perfect courses for any occasion. Just tell us what you're looking for!")
        st.success("For Your Courses", icon="💙")
