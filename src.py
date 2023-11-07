import pandas as pd
import numpy as np
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from googletrans import Translator
import requests
import re
from sklearn.metrics.pairwise import cosine_similarity

# Initialize NLTK (for English text processing)
nltk.download('punkt')
nltk.download('stopwords')

# Initialize Google Translator
translator = Translator()

# Load data from an Excel file containing English questions and answers
excel_file_path = r"C:\Users\Shrav\Dev\shetkariapp\agriculture_data.xlsx"
data = pd.read_excel(excel_file_path)

# Initialize English stopwords
english_stopwords = set(nltk.corpus.stopwords.words('english'))

# Define cities in Maharashtra
maharashtra_cities = ['mumbai', 'pune', 'thane', 'nagpur', 'nashik', 'aurangabad', 'amravati', 'solapur', 'kolhapur']

# Function to preprocess text for English
def preprocess_text_english(text):
    words = nltk.word_tokenize(text.lower())
    words = [word for word in words if word not in english_stopwords]
    return ' '.join(words)

# Function to extract entities using spaCy
def extract_entities(user_input, language):
    translated_text = translator.translate(user_input, src=language, dest='en').text
    entities = nltk.word_tokenize(translated_text.lower())  # Using NLTK for simplicity
    return entities

# Initialize a CountVectorizer and MultinomialNB
vectorizer = CountVectorizer()
classifier = MultinomialNB()

# Fit the vectorizer with the training data
X_train = [preprocess_text_english(question) for question in data['Question']]
X_train_vec = vectorizer.fit_transform(X_train)
y_train = data['Category']
classifier.fit(X_train_vec, y_train)

# Function to extract city and date from user input
def extract_city_and_date(user_input, language):
    entities = extract_entities(user_input, language)
    city = None
    date = None

    for entity in entities:
        if entity.lower() in maharashtra_cities:
            city = entity.lower()
        date_match = re.search(r'\b\d{1,2}-\d{1,2}-\d{4}\b', entity)
        if date_match:
            date = date_match.group()

    return city, date

# Function to get intent from user input using Naive Bayes
def get_intent_naive_bayes(user_input, language):
    user_input_translated = translator.translate(user_input, src=language, dest='en').text
    user_input_english = preprocess_text_english(user_input_translated)
    user_input_vec = vectorizer.transform([user_input_english])
    intent = classifier.predict(user_input_vec)[0]

    # Check for weather-related keywords along with city and date
    keywords = ['weather', 'temperature', 'rainfall']
    entities = extract_entities(user_input, language)
    city_detected = any(entity.lower() in maharashtra_cities for entity in entities)
    date_detected = any(re.search(r'\b\d{1,2}-\d{1,2}-\d{4}\b', entity) for entity in entities)
    intent_detected = any(keyword in entities for keyword in keywords)

    if city_detected and date_detected and intent_detected:
        intent = 'weather'

    return intent

# Function to get weather information
def get_weather(city, date):
    api_key = '6daa8091ea0a64e28c136f2a3a55a3b9'  # Replace with your actual API key
    url = f'http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}'
    response = requests.get(url)
    data = response.json()
    max_temp = data['main']['temp_max'] - 273.15
    min_temp = data['main']['temp_min'] - 273.15

    # Get other weather information
    humidity = data['main']['humidity']
    wind_speed = data['wind']['speed']
    sunrise_timestamp = data['sys']['sunrise']
    sunset_timestamp = data['sys']['sunset']

    # Convert sunrise and sunset times to readable format
    from datetime import datetime
    sunrise_time = datetime.fromtimestamp(sunrise_timestamp).strftime('%Y-%m-%d %H:%M:%S')
    sunset_time = datetime.fromtimestamp(sunset_timestamp).strftime('%Y-%m-%d %H:%M:%S')

    # Create formatted weather report
    report = f"Weather in {city} on {date}:\n"
    report += f"Max Temperature: {max_temp:.2f}°C\n"
    report += f"Min Temperature: {min_temp:.2f}°C\n"
    report += f"Humidity: {humidity}%\n"
    report += f"Wind Speed: {wind_speed} m/s\n"
    report += f"Sunrise Time: {sunrise_time}\n"
    report += f"Sunset Time: {sunset_time}\n"

    return report

# Function to get the response based on intent and extracted entities
def get_response_based_on_intent(intent, language, user_input, extracted_entities):
    relevant_responses = []
    if intent == 'weather':
        city, date = extract_city_and_date(user_input, language)
        if city and date:
            response = get_weather(city, date)
        else:
            response = "Sorry, I couldn't understand the city or date."
    else:
        user_input_translated = translator.translate(user_input, src=language, dest='en').text
        user_input_vec = vectorizer.transform([preprocess_text_english(user_input_translated)])
        for index, row in data.iterrows():
            question_vec = vectorizer.transform([preprocess_text_english(row['Question'])])
            similarity = cosine_similarity(user_input_vec, question_vec)
            if similarity > 0.5 and intent == row['Category']:
                relevant_responses.append(row['Answer'])
        if relevant_responses:
            response = np.random.choice(relevant_responses)
        else:
            response = "I'm not sure how to respond to that."

    return response

# Function to ask user for language preference
def get_language_preference():
    while True:
        preference = input("Please enter your language preference (English or Marathi): ").lower()
        if preference in ['english', 'marathi']:
            return preference
        else:
            print("Invalid choice. Please enter 'English' or 'Marathi'.")

# Main chat function
def chat():
    print("Welcome to the Agriculture Chatbot!")
    language_preference = get_language_preference()
    while True:
        user_input = input(f"You ({language_preference.capitalize()}): ")
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        intent = get_intent_naive_bayes(user_input, language_preference)
        print(intent)

        response = get_response_based_on_intent(intent, language_preference, user_input, [])

        response_translated = translator.translate(response, src='en', dest=language_preference).text
        print(f"Bot ({language_preference.capitalize()}):", response_translated)

if __name__ == '__main__':
    chat()
