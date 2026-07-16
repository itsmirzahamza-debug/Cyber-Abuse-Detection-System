import streamlit as st
import pickle
import re
import pandas as pd
import numpy as np
streamlit_app.py

# Page configuration
st.set_page_config(page_title="Cyber Abuse Detector", page_icon="🛡️", layout="wide")

# --- STEP 1: LOAD THE VECTORIZER AND MODELS (WITH CACHING) ---
@st.cache_resource
def load_assets():
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open('Naive_Bayes_model.pkl', 'rb') as f:
        nb_model = pickle.load(f)
    with open('SVM_model.pkl', 'rb') as f:
        svm_model = pickle.load(f)
    with open('Random_Forest_model.pkl', 'rb') as f:
        rf_model = pickle.load(f)
    
    return vectorizer, {
        "Naive Bayes": nb_model, 
        "SVM": svm_model, 
        "Random Forest": rf_model
    }

try:
    vectorizer, models = load_assets()
except FileNotFoundError:
    st.error("⚠️ Error: Model files (.pkl) or Vectorizer not found. Please ensure your saved model files are in the exact same directory as this script.")
    st.stop()

# --- STEP 2: PREPROCESSING FUNCTION ---
def clean_input_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text) 
    return " ".join(text.split())

# --- STEP 3: USER INTERFACE ---
st.title("🛡️ Roman Urdu Cyber Abuse Detector")
st.write("An end-to-end Machine Learning web application designed to detect toxic and abusive comments in Romanized Urdu/Hindi script.")
st.markdown("---")

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["💬 Single Comment Analysis", "📁 Bulk CSV Processing", "🏆 Model Leaderboard"])

# ==================== TAB 1: SINGLE COMMENT ANALYSIS ====================
with tab1:
    st.subheader("Analyze a Single Comment")
    user_input = st.text_area("✍️ Enter Roman Urdu/Hindi text here:", placeholder="Type a comment... (e.g., Tu bohot bekar aadmi hai)")
    
    if st.button("🔍 Check Comment", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter some text before analyzing.")
        else:
            cleaned_text = clean_input_text(user_input)
            vectorized_text = vectorizer.transform([cleaned_text])
            
            st.markdown("### 📊 Live Model Predictions")
            st.write(f"*Original Input:* \"{user_input}\"")
            st.markdown("---")
            
            col1, col2, col3 = st.columns(3)
            
            # --- Column 1: Naive Bayes ---
            with col1:
                st.markdown("#### 🧮 Naive Bayes")
                pred = models["Naive Bayes"].predict(vectorized_text)[0]
                prob = models["Naive Bayes"].predict_proba(vectorized_text)[0]
                confidence = max(prob) * 100
                
                if pred == 1:
                    st.error("🚨 **Abusive Detected**")
                else:
                    st.success("✅ **Clean Comment**")
                st.info(f"Confidence: **{confidence:.2f}%**")
                
            # --- Column 2: SVM ---
            with col2:
                st.markdown("#### 📈 SVM (Best Performer)")
                pred = models["SVM"].predict(vectorized_text)[0]
                prob = models["SVM"].predict_proba(vectorized_text)[0]
                confidence = max(prob) * 100
                
                if pred == 1:
                    st.error("🚨 **Abusive Detected**")
                else:
                    st.success("✅ **Clean Comment**")
                st.info(f"Confidence: **{confidence:.2f}%**")
                
            # --- Column 3: Random Forest ---
            with col3:
                st.markdown("#### 🌲 Random Forest")
                pred = models["Random Forest"].predict(vectorized_text)[0]
                prob = models["Random Forest"].predict_proba(vectorized_text)[0]
                confidence = max(prob) * 100
                
                if pred == 1:
                    st.error("🚨 **Abusive Detected**")
                else:
                    st.success("✅ **Clean Comment**")
                st.info(f"Confidence: **{confidence:.2f}%**")
            
            st.markdown("---")
            with st.expander("Show Technical Preprocessing Details"):
                st.write(f"**Cleaned Text (Input after removing punctuation and case normalization):** `{cleaned_text}`")


with tab3:
    st.subheader("🏆 Model Performance Leaderboard")
    st.write("Below are the statistical metrics generated during the training phase on the testing split of the dataset:")
    
    leaderboard_data = {
        "Model Name": ["Support Vector Machine (SVM)", "Random Forest Classifier", "Multinomial Naive Bayes"],
        "Accuracy Score": ["80.06%", "79.12%", "78.43%"],
        "Abuse (Class 1) F1-Score": ["0.78", "0.76", "0.74"],
        "Clean (Class 0) F1-Score": ["0.82", "0.81", "0.80"],
        "Computational Speed": ["Medium (Best Results)", "Slow", "Instant (Lightweight)"]
    }
    
    leaderboard_df = pd.DataFrame(leaderboard_data)
    st.table(leaderboard_df)
    
    st.info("💡 **Key Finding:** The Support Vector Machine (SVM) delivers the highest precision and F1-score for detecting abusive keywords in Roman Urdu, minimizing the number of False Negatives.")
