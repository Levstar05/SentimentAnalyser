import streamlit as st
from transformers import pipeline
import plotly.express as px
import pandas as pd

# Load pre-trained emotion detection pipeline with more emotions
emotion_pipeline = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", return_all_scores=True)

# Streamlit app interface with enhanced visuals
st.set_page_config(page_title="Emotion Detection App", page_icon="📝", layout="wide")
st.markdown("""
    <style>
        body {
            background-color: #f0f2f6;
            color: #333333;
        }
        .stTextArea textarea {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            border-radius: 10px;
            padding: 10px 20px;
            font-size: 16px;
        }
        .stButton button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🌟 Advanced Emotion Detection App")
st.write("💬 Enter text below to analyze its emotional tone and get a detailed breakdown.")

# User input
user_input = st.text_area("✍️ Type your text here:", height=150)

# Analyze emotions when button is clicked
if st.button("🚀 Analyze Emotions"):
    if user_input.strip():
        with st.spinner("🔍 Analyzing emotions..."):
            results = emotion_pipeline(user_input)[0]

        # Display top emotion
        top_emotion = max(results, key=lambda x: x['score'])
        st.success(f"🎭 **Dominant Emotion:** {top_emotion['label']} ({top_emotion['score']:.2f})")

        # Data for visualization
        labels = [res['label'] for res in results]
        scores = [res['score'] for res in results]
        df = pd.DataFrame({"Emotion": labels, "Confidence": scores})

        # Advanced Breakdown: Interactive Plotly chart
        st.subheader("📊 Emotion Breakdown")
        fig = px.bar(df, x="Emotion", y="Confidence", color="Emotion",
                     color_discrete_sequence=px.colors.qualitative.Pastel,
                     title="Confidence Levels for Detected Emotions")
        fig.update_layout(xaxis_title="Emotion", yaxis_title="Confidence",
                          template="plotly_white", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Detailed feedback
        st.subheader("📄 Detailed Emotion Scores")
        for res in results:
            st.write(f"**{res['label']}:** {res['score']:.2f}")
    else:
        st.warning("⚠️ Please enter some text before analyzing.")

# Footer
st.markdown("""
    <hr>
    <p style='text-align: center; color: gray;'>
    Made with ❤️ using Hugging Face, Streamlit, and Plotly
    </p>
""", unsafe_allow_html=True)
