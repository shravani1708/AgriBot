from flask import Flask, render_template, request, session, redirect, url_for
from src import get_intent_naive_bayes, get_response_based_on_intent
from googletrans import Translator

translator = Translator()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this to a random secret key

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_language_preference', methods=['POST'])
def set_language_preference():
    language_preference = request.form['language_preference']
    session['language_preference'] = language_preference
    return {'success': True}

@app.route('/chat')
def chat():
    if 'language_preference' in session:
        return render_template('chat.html', language_preference=session['language_preference'])
    else:
        return redirect(url_for('index'))

@app.route('/get_response', methods=['POST'])
def get_response():
    user_input = request.form['user_input']
    language_preference = session['language_preference']
    intent = get_intent_naive_bayes(user_input, language_preference)
    response = get_response_based_on_intent(intent, language_preference, user_input, [])
    translated_response = translator.translate(response, src='en', dest=language_preference).text
    return {'response': translated_response}

if __name__ == '__main__':
    app.run(debug=True)
