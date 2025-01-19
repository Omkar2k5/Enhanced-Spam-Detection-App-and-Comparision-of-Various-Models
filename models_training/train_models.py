import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import os


# Preprocessing Function
def preprocess_text(text, method="none"):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    if method == "stemming":
        stemmer = PorterStemmer()
        words = [stemmer.stem(word) for word in words if word not in stopwords.words("english")]
    elif method == "lemmatization":
        lemmatizer = WordNetLemmatizer()
        words = [lemmatizer.lemmatize(word) for word in words if word not in stopwords.words("english")]
    return " ".join(words)


# Load Dataset
data_path = "C:\\Intership Technook\\Project\\data\\spam.csv"
data = pd.read_csv(data_path, encoding="latin-1")

data = data.rename(columns={"v1": "label", "v2": "message"})[["label", "message"]]
data["label"] = data["label"].map({"ham": 0, "spam": 1})
data["processed_message"] = data["message"].apply(preprocess_text)

X = data["processed_message"]
y = data["label"]

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Directories if They Don't Exist
os.makedirs("../vectorizers", exist_ok=True)
os.makedirs("../models", exist_ok=True)
os.makedirs("../data", exist_ok=True)

# Train Models and Save Them
vectorizers = {
    "count": CountVectorizer(),
    "tfidf": TfidfVectorizer()
}
models = {
    "naive_bayes": MultinomialNB(),
    "logistic_regression": LogisticRegression(),
    "random_forest": RandomForestClassifier(),
    "svm": SVC()
}

for vec_name, vectorizer in vectorizers.items():
    X_train_vectorized = vectorizer.fit_transform(X_train)
    joblib.dump(vectorizer, f"../vectorizers/{vec_name}.pkl")

    for model_name, model in models.items():
        model.fit(X_train_vectorized, y_train)
        joblib.dump(model, f"../models/{model_name}_{vec_name}.pkl")

# Save Preprocessed Test Data
test_data = pd.DataFrame({"processed_message": X_test, "label": y_test})
test_data.to_csv("../data/preprocessed_test_data.csv", index=False)
