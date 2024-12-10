import streamlit as st

st.set_page_config(page_title="Courses Recommender", page_icon="💻", layout="centered")         

from streamlit_pills import pills
import tensorflow as tf
import pandas as pd
from main import recommend_courses, sidebar, tfidf_matrix, df
from sklearn.feature_extraction.text import TfidfVectorizer


sidebar()

col1, col2 = st.columns([1, 4])  

with col1:
    st.image('logo/12.png', use_column_width=True)  

with col2:
    st.title('Course Recommendation System')  

st.header('About this app')

st.subheader('🤖What can this app do?')
st.info(
    'Course recommender was designed to help users find the most relevant online courses that match their interests and needs, contained more that 50 IT Courses. '
    'This app uses a Neural Collaborative Filtering (NCF) and Content Based model to Built with the Coursera dataset, it provides personalized recommendations '
    'based on users past interactions and course preferences.'
)

st.subheader('📝How to use the app?')
st.warning(
    "To use the app, simply enter at least one keyword related to the type of course you're interested in (e.g., 'machine learning')"
    "in the text box provided. When you click 'Get Recommendations,' the app will analyze your input and suggest the most relevant courses based on similarity to other user interactions. "
    "You’ll receive a list of recommended courses with links to access them."
)

st.subheader('📁Datasets')
st.text('Coursera Dataset from Kaggle')

selected = pills("Eg.", ["Machine Learning for Beginner", "artificial intelligence courses (ai)", "Software Product Management"], ["🤖", "💻","📊"],clearable=True,index=None)

input_text = st.text_input('Enter a keyword or course interest description:',selected)


if st.button('Get Recommendation'):
    if input_text:
        st.write(f"### Recommendations for {input_text}")
        
        # Mendapatkan rekomendasi
        recommendations = recommend_courses(input_text, tfidf_matrix, df)
        
        # Menambahkan kolom dengan link yang dapat diklik
        recommendations['course_url_clickable'] = recommendations['course_url'].apply(
            lambda x: f"[Link]({x})"
        )
        
        # Menampilkan dataframe dengan kolom yang relevan
        st.dataframe(
            recommendations[['name', 'rating', 'course_url_clickable']].rename(
                columns={
                    'name': 'Course Name',
                    'rating': 'Rating',
                    'course_url_clickable': 'Course URL'
                }
            ),
            use_container_width=True
        )
    else:
        st.error("Harap masukkan kata kunci atau deskripsi minat.")
