from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

app = Flask(__name__)

def preprocess_text(text):
    return text.lower() if isinstance(text, str) else ""

def train_chatbot(dataset_path):
    df = pd.read_csv(dataset_path)
    df['questions'] = df['questions'].fillna("")
    df['questions'] = df['questions'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['questions'].values.astype('U'))

    classifier = MultinomialNB()
    classifier.fit(tfidf_matrix, df['category'])

    return df, vectorizer, classifier

def get_response(user_input, df, vectorizer, classifier, keyword_mapping):
    user_input = preprocess_text(user_input)

    # Menangani salam atau kata kunci khusus
    if user_input.lower() in ["hai", "halo", "hai!", "halo!", "Hai", "Halo", "Hai!", "Halo!", "hai chatbot", "halo chatbot", "hai chatbot!", "halo chatbot!", "Hai chatbot", "Halo chatbot", "Hai chatbot!", "Halo chatbot!" ]:
        return "Halo! Ada yang bisa saya bantu?"

    input_tokens = user_input.split()

    category_keywords = {category: keywords for category, keywords in keyword_mapping.items()}

    matching_keywords = {}
    matching_categories = []

    for category, keywords in category_keywords.items():
        intersection = set(input_tokens) & set(keywords)
        if intersection:
            matching_keywords[category] = list(intersection)
            matching_categories.append(category)

    if not matching_categories:
        predicted_category = classifier.predict(vectorizer.transform([user_input]))
        matching_categories.append(predicted_category[0])

    input_vector = vectorizer.transform([user_input])
    filtered_df = df[df['category'].isin(matching_categories)]

    if not filtered_df.empty:
        tfidf_matrix_filtered = vectorizer.transform(filtered_df['questions'].values.astype('U'))
        cosine_similarities = cosine_similarity(input_vector, tfidf_matrix_filtered).flatten()
        max_similarity_index = cosine_similarities.argmax()
        response = filtered_df['answer'].iloc[max_similarity_index]
    else:
        response = "Maaf, saya tidak mengerti pertanyaan Anda."

    return response


dataset_path = 'knowledgebase.csv'
keyword_mapping = {
    'Category1': ['keyword1', 'keyword2'],
    'Category2': ['keyword3', 'keyword4'],
    # ...
}

df, vectorizer, classifier = train_chatbot(dataset_path)

@app.route('/')
def index():
    welcome_message = "Halo! Senang berjumpa dengan Anda!"
    bot_response = get_response(welcome_message, df, vectorizer, classifier, keyword_mapping)
    return render_template('index.html', welcome_message=welcome_message, bot_response=bot_response)

@app.route('/get_response', methods=['POST'])
def get_bot_response():
    user_input = request.form['user_input']
    response = get_response(user_input, df, vectorizer, classifier, keyword_mapping)
    return jsonify({'response': response})

if __name__ == "__main__":
    app.run(debug=True)