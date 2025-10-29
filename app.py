from  flask import Flask, render_template, request
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import sent_tokenize
import nltk

# Download the punkt tokenizer from NLTK (only needed once)
nltk.download('punkt')

app = Flask(__name__)

def summarize_text(text, num_sentences=3):
    # Split the text into sentences
    sentences = sent_tokenize(text)

    # Create a TF-IDF Vectorizer
    tfidf = TfidfVectorizer()

    # Transform the sentences into TF-IDF features
    tfidf_matrix = tfidf.fit_transform(sentences)

    # Sum the TF-IDF scores for each sentence
    sentence_scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

    # Get the indices of the top N sentences
    top_sentence_indices = sentence_scores.argsort()[-num_sentences:][::-1]

    # Extract the top sentences
    summary_sentences = [sentences[i] for i in sorted(top_sentence_indices)]

    return ' '.join(summary_sentences)

@app.route('/', methods=['GET', 'POST'])
def index():
    summary = ""
    if request.method == 'POST':
        text = request.form['text']
        num_sentences = int(request.form.get('num_sentences', 3))
        summary = summarize_text(text, num_sentences)
    return render_template('index.html', summary=summary)

if __name__ == '__main__':
    app.run(debug=True)
