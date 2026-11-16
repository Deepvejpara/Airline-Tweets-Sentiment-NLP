"""Streamlit web app for Airline Tweet Sentiment Analysis (Step 6)."""

import joblib
import pandas as pd
import streamlit as st

from nlp_helpers import (
    clean_text,
    extract_mentions_and_hashtags,
    get_pos_tags,
)

st.set_page_config(
    page_title="Airline Tweet Sentiment Analyzer",
    page_icon="✈️",
    layout="centered",
)

LABEL_MAP = {
    0: "Negative 😠",
    1: "Neutral 😐",
    2: "Positive 😊",
}

SUGGESTED_ACTIONS = {
    0: "Route to customer care for priority follow-up.",
    1: "Log feedback and monitor for any follow-up reply.",
    2: "Share positive feedback with the crew and social media team.",
}

EXAMPLE_TWEETS = {
    "Negative": "@IndiGo6E flight 6E-204 delayed by 4 hours and my bag is still missing! #worst #delayed",
    "Neutral": "@AirIndiaX checking if flight AI-102 departs from Terminal 3 today? #travel",
    "Positive": "@Vistara thank you for the smooth boarding and super helpful cabin crew! #greatservice",
}


@st.cache_resource
def load_sentiment_pipeline():
    """Load the trained scikit-learn sentiment pipeline from disk."""
    return joblib.load("sentiment_pipeline.pkl")


st.title("✈️ Airline Tweet Sentiment Analyzer")
st.write(
    "Paste a passenger tweet below to classify its sentiment (Negative, Neutral, or Positive) "
    "and inspect extracted entities."
)

with st.sidebar:
    st.header("Example Tweets")
    for sentiment_name, example in EXAMPLE_TWEETS.items():
        if st.button(f"Load {sentiment_name} Example", use_container_width=True):
            st.session_state["tweet_input"] = example

    st.divider()
    st.subheader("About this project")
    st.write(
        "Built for the NLP Practical Exam (Set B). Uses RegEx cleaning, NLTK tokenisation "
        "and lemmatization (preserving negations like 'not', 'no', 'nor'), and a TF-IDF + "
        "Logistic Regression pipeline trained on the Twitter US Airline Sentiment dataset."
    )

tweet_text = st.text_area(
    "Passenger Tweet",
    key="tweet_input",
    height=120,
    placeholder="Paste a tweet here, e.g. @Airline flight delayed again! #frustrated",
)

if st.button("Analyse", type="primary"):
    if not tweet_text or not tweet_text.strip():
        st.warning("Please enter a tweet")
    else:
        try:
            pipeline = load_sentiment_pipeline()
        except FileNotFoundError:
            st.error(
                "Model file 'sentiment_pipeline.pkl' not found. "
                "Please run NLP_Practical_Exam_SetB.ipynb (Step 4.6) first."
            )
            st.stop()

        cleaned = clean_text(tweet_text)
        pred_class = int(pipeline.predict([cleaned])[0])
        probs = pipeline.predict_proba([cleaned])[0]

        st.subheader(f"Predicted Sentiment: {LABEL_MAP[pred_class]}")
        st.info(f"**Suggested action:** {SUGGESTED_ACTIONS[pred_class]}")

        prob_df = pd.DataFrame(
            {"Probability": [float(probs[0]), float(probs[1]), float(probs[2])]},
            index=["Negative", "Neutral", "Positive"],
        )
        st.write("**Class Probabilities**")
        st.bar_chart(prob_df)

        mentions, hashtags = extract_mentions_and_hashtags(tweet_text)
        col1, col2 = st.columns(2)
        with col1:
            st.write("**@Mentions found**")
            st.write(", ".join(mentions) if mentions else "None")
        with col2:
            st.write("**#Hashtags found**")
            st.write(", ".join(hashtags) if hashtags else "None")

        with st.expander("View POS Tags (Bonus)"):
            pos_rows = get_pos_tags(tweet_text)
            if pos_rows:
                pos_df = pd.DataFrame(pos_rows, columns=["Word", "POS Tag"])
                st.dataframe(pos_df, use_container_width=True, hide_index=True)
            else:
                st.write("No tokens found.")