from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load the saved model and tokenizer
model = load_model("text_generation_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Define max_sequence_len (same as during model training)
max_sequence_len = 100


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    seed_text = request.form.get('prompt')
    next_words = int(request.form.get('next_words', 2))

    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

        predictions = model.predict(token_list, verbose=0)
        predicted = np.argmax(predictions, axis=-1)[0]

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return jsonify({"generated_text": seed_text})


@app.route('/suggest', methods=['GET'])
def suggest():
    query = request.args.get('q', '')

    if not query:
        return jsonify([])

    # Generate sentence suggestions based on partial input
    suggestions = []

    for word in tokenizer.word_index.keys():
        if word.startswith(query.lower()):
            suggestions.append(word)

    return jsonify(suggestions[:10])


if __name__ == '__main__':
    app.run(debug=True)
