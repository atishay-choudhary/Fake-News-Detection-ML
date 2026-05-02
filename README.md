# 📰 Fake News Detection using Machine Learning

An end-to-end Machine Learning application that classifies news articles as **Fake** or **Real** using Natural Language Processing (NLP) techniques.

📌 Note
This project is developed as part of a Machine Learning course to demonstrate a complete end-to-end ML workflow, including preprocessing, modeling, evaluation, and deployment.

---

## 📌 Problem Statement

The objective of this project is to detect fake news using textual data. The problem is formulated as a **binary classification task**, where news articles are classified as:

- **0 → Fake News**
- **1 → Real News**

With the rise of misinformation, automated detection systems are essential to identify unreliable content efficiently.

---

## ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas & NumPy
- TF-IDF (Text Feature Extraction)
- Matplotlib & Seaborn (EDA)
- WordCloud (Text Visualization)
- Streamlit (Web Application)

---

## 📊 Workflow

1. Data Collection & Merging
2. Data Cleaning & Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Extraction using TF-IDF
5. Model Training & Evaluation
6. Model Optimization (Regularization)
7. Unsupervised Learning (PCA, K-Means)
8. Advanced Models (Decision Tree, Random Forest)
9. Model Comparison & Selection
10. Deployment using Streamlit

---

## 🤖 Models Implemented

- Logistic Regression (Baseline Model)
- Decision Tree
- Random Forest (Final Model)

---

## 📈 Model Performance

| Model               | Accuracy         |
| ------------------- | ---------------- |
| Logistic Regression | 82.8%            |
| Decision Tree       | 82.1%            |
| Random Forest       | **82.9% (Best)** |

---

## 🚀 Features

- Text-based prediction
- File upload support (.txt)
- Confidence score display
- Probability breakdown (Fake vs Real)
- Clean and interactive UI

---

## 🧠 Key Insights

- Ensemble models (Random Forest) provide better generalization
- TF-IDF is effective for high-dimensional text classification
- Model performance is stable across multiple evaluation metrics

---

## ⚠️ Limitations

- Model detects **writing patterns**, not factual correctness
- Dataset bias (political/news domain) affects generalization
- May misclassify unseen writing styles

---

## 📁 Project Structure

Course Project
|-> app.py
|-> Fake.csv
|-> FakeNewsDetection.ipynb
|-> fake_news_dataset.csv
|-> generative_ai_misinformation_dataset.csv
|-> model.pkl
|-> readme.md
|-> requirements.txt
|-> True.csv
|-> vectorizer.pkl

## ▶️ How to Run Locally

> pip install -r requirements.txt
> streamlit run app.py

## Deployment

- The application is deployed using Streamlit Cloud.

## Future Scope

- Use deep learning models (LSTM, BERT)
- Integrate fact-checking APIs
- Expand dataset across multiple domains
- Build hybrid models (text + metadata)

# Author

- Atishay Choudhary.
