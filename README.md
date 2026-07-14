# Cyber-Abuse-Detection-System
Roman Urdu Cyber Abuse Detector  An end-to-end Machine Learning web application designed to detect toxic and abusive comments in Romanized Urdu/Hindi script. This tool leverages multiple NLP classification models to identify cyberbullying, offensive words, and toxic behavior.
# 🛡️ Roman Urdu Cyber Abuse Detector

An end-to-end Machine Learning web application designed to detect toxic and abusive comments in Romanized Urdu/Hindi script. This tool leverages multiple NLP classification models to identify cyberbullying, offensive words, and toxic behavior.

## 🚀 Live Streamlit Application Features
* **Single Comment Analysis:** Type any comment in Roman Urdu to instantly analyze it with confidence scores (probabilities).
* **Multi-Model Predictions:** Compares **SVM (Linear)**, **Multinomial Naive Bayes**, and **Random Forest** side-by-side.
* **Bulk CSV Processing:** Upload a list of comments in a `.csv` file, run bulk moderation, and download annotated results instantly.
* **Model Leaderboard:** View static training performance metrics of all implemented algorithms.

---

## 🛠️ System Architecture & Working Mechanism

1. **Text Preprocessing:** Case normalization, removal of punctuation/special characters, and whitespace tokenization.
2. **Feature Extraction:** Standardized TF-IDF (Term Frequency-Inverse Document Frequency) Vectorization capturing unigrams and bigrams.
3. **Classification Classifiers:**
   * **Support Vector Machine (SVM):** Best performing model with **80.06% Accuracy** ($F1$-score: $0.78$).
   * **Random Forest Classifier:** Robust ensemble method (~79.12% Accuracy).
   * **Multinomial Naive Bayes (NB):** Lightweight, ultra-fast probabilistic baseline model (~78.43% Accuracy).

---

## 💻 How to Run Locally
http://localhost:8501/
### 1. Clone this repository
```bash
git clone [https://github.com/YOUR_USERNAME/roman-urdu-abuse-detector.git](https://github.com/itsmirzahamza/roman-urdu-abuse-detector.git)
cd roman-urdu-abuse-detector
