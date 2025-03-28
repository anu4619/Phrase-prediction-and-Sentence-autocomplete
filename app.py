from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

app = Flask(__name__)

# Load your saved model
model = load_model("text_generation_model.keras")

# Load the tokenizer
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
    next_words = int(request.form.get('next_words', 10))  # Default to 10 words
    num_results = int(request.form.get('num_results', 3))  # Generate 3 results by default

    results = []

    for _ in range(num_results):
        current_text = seed_text
        for _ in range(next_words):
            token_list = tokenizer.texts_to_sequences([current_text])[0]
            token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')

            predictions = model.predict(token_list, verbose=0)
            predicted = np.argmax(predictions, axis=-1)[0]

            output_word = ""
            for word, index in tokenizer.word_index.items():
                if index == predicted:
                    output_word = word
                    break

            current_text += " " + output_word

        results.append(current_text)

    return jsonify({"generated_texts": results})


if __name__ == '__main__':
    app.run(debug=True)
