import streamlit as st
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
import joblib
import os
from model_charts import plot_accuracy_comparison, plot_confusion_matrix, plot_classification_report, classification_report_as_dataframe

st.set_page_config(page_title="Enhanced Spam Detection App", page_icon="📧", layout="wide")

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')

# Preload models and vectorizers
@st.cache_resource
class ModelStore:
    def __init__(self):
        self.models = {}

    def get(self, preprocessing, vectorizer, model_type):
        return self.models.get((preprocessing, vectorizer, model_type))

    def add(self, preprocessing, vectorizer, model_type, model, vectorizer_obj):
        self.models[(preprocessing, vectorizer, model_type)] = (model, vectorizer_obj)

model_store = ModelStore()

def load_pretrained_components(model_type, vectorizer_type):
    vectorizer_path = f"C:/Intership Technook/Project/vectorizers/{vectorizer_type}.pkl"
    model_path = f"C:/Intership Technook/Project/models/{model_type}_{vectorizer_type}.pkl"

    if os.path.exists(vectorizer_path) and os.path.exists(model_path):
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        return model, vectorizer
    else:
        st.error(f"Model or Vectorizer not found for {model_type} with {vectorizer_type}!")
        return None, None

def load_data():
    data = pd.read_csv("C:\\Intership Technook\\Project\\data\\spam.csv", encoding="latin-1")
    data = data.rename(columns={'v1': 'label', 'v2': 'message'})
    data = data[['label', 'message']]
    data['label'] = data['label'].map({'ham': 0, 'spam': 1})
    return data

def preprocess_text(text, method="none"):
    if pd.isnull(text):
        return ""
    if not isinstance(text, str):
        text = str(text)

    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()

    if method == "stemming":
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words if word not in stopwords.words("english")]
    elif method == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")]
    elif method == "stopwords":
        words = [word for word in words if word not in stopwords.words("english")]

    return " ".join(words)

def train_and_store_model(preprocessing="none", vectorizer="count", model_type="naive_bayes"):
    data = load_data()
    data['processed_message'] = data['message'].apply(lambda x: preprocess_text(x, method=preprocessing))

    X = data['processed_message']
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if vectorizer == "count":
        vectorizer_obj = CountVectorizer()
    elif vectorizer == "tfidf":
        vectorizer_obj = TfidfVectorizer()

    X_train_vectorized = vectorizer_obj.fit_transform(X_train)
    X_test_vectorized = vectorizer_obj.transform(X_test)

    if model_type == "naive_bayes":
        model = MultinomialNB()
    elif model_type == "logistic_regression":
        model = LogisticRegression()
    elif model_type == "random_forest":
        model = RandomForestClassifier()
    elif model_type == "svm":
        model = SVC()

    model.fit(X_train_vectorized, y_train)
    model_store.add(preprocessing, vectorizer, model_type, model, vectorizer_obj)

    return model, vectorizer_obj, X_test_vectorized, y_test

def classification_report_as_dataframe(y_test, y_pred):
    report = classification_report(y_test, y_pred, output_dict=True)
    return pd.DataFrame(report).transpose()

# Main application
def main():
    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Menu", ["Home", "Predict", "Model Evaluation", "About"])

    if menu == "Home":
        st.title("📧 Enhanced Spam Detection App")
        st.markdown("""
        This app allows you to classify SMS messages as Spam or Ham, practice various NLP techniques, and benchmark different models.
        - **Preprocessing Options**: Stemming, Lemmatization, Stopword Removal
        - **Models**: Naive Bayes, Logistic Regression, Random Forest, SVM
        """)

    elif menu == "Predict":
        st.title("📩 Spam or Ham Prediction")
        vectorizer_type = st.selectbox("Select Vectorizer", ["count", "tfidf"])
        model_type = st.selectbox("Select Model", ["naive_bayes", "logistic_regression", "random_forest", "svm"])

        user_message = st.text_area("Type your SMS message here:")
        if st.button("Classify"):
            if not user_message.strip():
                st.warning("⚠ Please enter a valid message!")
            else:
                model, vectorizer = load_pretrained_components(model_type, vectorizer_type)
                processed_message = preprocess_text(user_message)
                message_vectorized = vectorizer.transform([processed_message])
                prediction = model.predict(message_vectorized)[0]
                label = "Spam" if prediction == 1 else "Ham"
                st.success(f"✅ The message is classified as: *{label}*")

    elif menu == "Model Evaluation":
        st.title("📊 Model Evaluation")

        vectorizer_type = st.selectbox("Select Vectorizer for Evaluation", ["count", "tfidf"])
        model_type = st.selectbox("Select Model for Evaluation",
                                  ["naive_bayes", "logistic_regression", "random_forest", "svm"])

        model, vectorizer = load_pretrained_components(model_type, vectorizer_type)
        test_data = load_data()
        test_data['processed_message'] = test_data['message'].apply(lambda x: preprocess_text(x, method="none"))
        X_test = vectorizer.transform(test_data["processed_message"])
        y_test = test_data["label"]

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Model Accuracy", f"{accuracy:.2f}")

        st.subheader("Confusion Matrix")

        confusion = confusion_matrix(y_test, y_pred)

        # Use Plotly to create a visually interactive confusion matrix
        fig = px.imshow(confusion,
                        labels=dict(x="Predicted", y="Actual"),
                        x=["Predicted Ham", "Predicted Spam"],
                        y=["Actual Ham", "Actual Spam"],
                        color_continuous_scale="Blues",
                        title=f"Confusion Matrix for {model_type}",
                        text_auto=True)
        st.plotly_chart(fig)

        # Plot comparison of model accuracies (you can replace with actual values)
        model_accuracies = {"Naive Bayes": 0.88, "Logistic Regression": 0.85, "Random Forest": 0.87,
                            "SVM": 0.90}  # Example accuracies
        fig = px.bar(x=list(model_accuracies.keys()), y=list(model_accuracies.values()),
                     labels={'x': 'Model', 'y': 'Accuracy'},
                     title="Model Accuracy Comparison",
                     color=list(model_accuracies.values()))
        st.plotly_chart(fig)



    elif menu == "About":
        st.title("About the App")
        st.markdown("""
        ## Welcome to the Enhanced Spam Detection App! 🚀
        This application is designed to classify SMS messages into **Spam** or **Ham** using various Natural Language Processing (NLP) techniques and machine learning models.  
        ### Key Features:
        - **Preprocessing Techniques**: Experiment with stemming, lemmatization, and stopword removal to optimize your model.
        - **Multiple Models**: Compare performance across Naive Bayes, Logistic Regression, Random Forest, and SVM.
        - **Interactive Analysis**: Visualize classification reports, confusion matrices, and model accuracy.
        - **Easy-to-Use Interface**: Designed for both beginners and experts in machine learning.
        ### About the Developer:
        Hi, I'm **Omkar Gondkar**! 👋
        - 🎓 Passionate about machine learning, NLP, and creating impactful applications.
        - 📧 Contact me: [gondkaromkar53@gmail.com](mailto:gondkaromkar53@gmail.com)
        - 🌐 [Connect on LinkedIn](https://www.linkedin.com/in/og25)
        - 💻 [Check out my GitHub](https://github.com/Omkar2k5)
        ### How It Works:
        1. Navigate to the **Predict** section to classify your messages.
        2. Explore the **Model Evaluation** section for performance metrics.
        3. Discover the various NLP techniques and models used under the hood.
        ### Feedback:
        We'd love to hear your feedback! Feel free to share your thoughts, suggestions, or questions on my gmail.
        Thank you for visiting, and happy exploring! 😊
        """)

if __name__ == "__main__":
    main()
