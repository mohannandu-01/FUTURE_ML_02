import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

nltk.download('stopwords')

# Load dataset
df = pd.read_csv("tickets.csv")

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    words = text.split()
    words = [word for word in words if word not in stopwords.words('english')]
    return " ".join(words)

df['cleaned'] = df['text'].apply(clean_text)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['cleaned'], df['category'], test_size=0.2)

# Build model pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('clf', LogisticRegression())
])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Category Classification Report:")
print(classification_report(y_test, y_pred))

# -------- PRIORITY LOGIC --------
def assign_priority(text):
    text = text.lower()
    if "not working" in text or "refund" in text or "charged" in text:
        return "high"
    elif "slow" in text or "issue" in text:
        return "medium"
    else:
        return "low"

# -------- TEST SYSTEM --------
while True:
    user_input = input("\nEnter customer ticket: ")
    if user_input == "exit":
        break

    cleaned = clean_text(user_input)
    category = model.predict([cleaned])[0]
    priority = assign_priority(user_input)

    print(f"\nCategory: {category}")
    print(f"Priority: {priority}")
