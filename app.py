import streamlit as st
import joblib
import numpy as np

# Load Model and Vectorizer
model = joblib.load("emotion_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Page Config
st.set_page_config(
    page_title="Emotion Predictor",
    page_icon="😊",
    layout="centered"
)

# CSS Styling
st.markdown("""
<style>
.title{
    text-align:center;
    font-size:40px;
    font-weight:bold;
    color:#4F46E5;
}
.subtitle{
    text-align:center;
    color:gray;
    font-size:18px;
}
.result{
    padding:20px;
    border-radius:10px;
    text-align:center;
    background-color:#f0f2f6;
    margin-top:20px;
}
</style>
""", unsafe_allow_html=True)

# Title
st.markdown("<div class='title'>😊 Emotion Predictor</div>",
            unsafe_allow_html=True)

st.markdown("<div class='subtitle'>Predict emotions from text using NLP</div>",
            unsafe_allow_html=True)

st.write("")

# Text Input
user_text = st.text_area(
    "Enter Text",
    placeholder="Type something here..."
)

# Emotion Mapping
emotion_map = {
    0: "sadness",
    1: "joy",
    2: "love",
    3: "anger",
    4: "fear",
    5: "surprise"
}

# Emoji Mapping
emotion_emoji = {
    "joy": "😄",
    "happy": "😊",
    "sadness": "😢",
    "sad": "😔",
    "anger": "😡",
    "fear": "😨",
    "love": "❤️",
    "surprise": "😲",
    "disgust": "🤢"
}

# Predict Button
if st.button("Predict Emotion"):

    if user_text.strip() == "":
        st.warning("Please enter some text.")
    else:

        # Convert text to vector
        text_vector = vectorizer.transform([user_text])

        # Prediction
        prediction = model.predict(text_vector)[0]

        # Convert prediction to emotion
        if isinstance(prediction, (np.integer, int)):
            emotion = emotion_map.get(int(prediction), "Unknown")
        else:
            emotion = str(prediction)

        # Emoji
        emoji = emotion_emoji.get(emotion.lower(), "🙂")

        # Display Result
        st.markdown(
            f"""
            <div class='result'>
                <h1>{emoji}</h1>
                <h2>Predicted Emotion</h2>
                <h2>{emotion.upper()}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Confidence Score
        try:
            confidence = np.max(model.predict_proba(text_vector)) * 100

            st.write("")
            st.progress(int(confidence))
            st.success(f"Confidence: {confidence:.2f}%")

        except:
            pass