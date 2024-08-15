import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from flask import Flask, render_template, request, jsonify, url_for
import joblib
import os
from sklearn.svm import SVC
# Load data
data = pd.read_csv('D:/IID2/S2/BIGData/Projet/archive (1)/train.csv', encoding="ISO-8859-1")

# Handle missing values
data['selected_text'].fillna('', inplace=True)

# Encode sentiments
sentiment_map = {"positive": 1, "neutral": 0, "negative": 2}
data['sentiment'] = data['sentiment'].map(sentiment_map).astype(int)

# Split data into features and labels
X = data['selected_text']
y = data['sentiment']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Vectorize text data
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define Naive Bayes classifier
nb = MultinomialNB()

# Train the model
nb.fit(X_train_vec, y_train)

# Save the vectorizer and classifier
joblib.dump(vectorizer, 'vectorizer.pkl')
joblib.dump(nb, 'svm1.pkl')

# Define paths to images
image_dir = "image"  

# Create Flask app
app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input
        data = request.form.get('selected_text')
        if not data:
            raise ValueError("No input text provided")

        # Vectorize input data
        data_vec = vectorizer.transform([data])

        # Perform prediction
        prediction = nb.predict(data_vec)[0]
        confidence = nb.predict_proba(data_vec).max()

        # Translate prediction to sentiment string
        sentiment_dict = {0: {"label": "Neutral", "emoji": "neutre.png"},
                          1: {"label": "Positive", "emoji":"smile.png"},
                          2: {"label": "Negative", "emoji": "sad.png"}}
        result = sentiment_dict[prediction]

        # Get overall accuracy
        y_pred = nb.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)

        # Construct response data
        response_data = {
            'sentiment': result['label'],
            'emoji': url_for('static', filename=f"{image_dir}/{result['emoji']}"),
            'accuracy': accuracy,
            'confidence': confidence
        }

        # Return result with accuracy
        return jsonify(response_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
