import streamlit as st
import joblib
import preprocess from preprocessing
model1 = joblib.load("lr_model.pkl")
model2 = joblib.load("mnb_model.pkl")

vectorizer = joblib.load("tfidf_vectorizer.pkl")

st.title("IMDb Review Sentiment Analyzer")

user_input = st.text_area("Enter a movie review:")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter a review.")
    else:
        vectorized_input = vectorizer.transform([preprocess(user_input)])
        prediction1 = model1.predict(vectorized_input)[0]
        st.success(f"Sentiment predicted by logistic regression: **{prediction1.capitalize()}**")
        prediction2 = model2.predict(vectorized_input)[0]
        st.success(f"Sentiment predicted by naive bayes: **{prediction2.capitalize()}**")