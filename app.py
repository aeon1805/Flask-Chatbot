import requests
from flask import Flask, render_template, request, jsonify
from nltk.chat.util import Chat, reflections
from transformers import pipeline
from datetime import datetime
import pytz
import re
import os
app = Flask(__name__)
emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", return_all_scores=True)

pairs = [
    ['my name is (.*)', ['Hi %1!']],
    ['(hi|hello|hey|hola)', ['Hey there!', 'Hi there!', 'Hello!']],
    ['(.*) in (.*) is fun', ['%1 in %2 is indeed fun!']],
    ['(.*)(location|city)', ['Mumbai, India']],
    ['(.*) created you?', ['Advaith did, using NLTK!']],
    ['(.*)help(.*)', ['I can help you with weather, time, or a quick chat.']],
    ['(.*) your name?', ['My name is A.D.I.']]
]
chat = Chat(pairs, reflections)
HF_TOKEN = os.getenv("HF_TOKEN") or "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"
def detect_emotion(text):
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}"
    }
    API_URL = "https://api-inference.huggingface.co/models/j-hartmann/emotion-english-distilroberta-base"
    payload = {"inputs": text}
    response = requests.post(API_URL, headers=headers, json=payload)
    results = response.json()[0]
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[0]["label"], results[0]["score"]

def get_weather(city):
    api_key = os.environ.get("OPENWEATHERMAP_KEY")
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url).json()
    if response.get("cod") != 200:
        return "Sorry, I couldn't fetch the weather right now."
    temp = response["main"]["temp"]
    desc = response["weather"][0]["description"]
    return f"The weather in {city.title()} is {desc} with a temperature of {temp}°C."

def get_time(city):
    timezones = {
        "new york": "America/New_York",
        "mumbai": "Asia/Kolkata",
        "tokyo": "Asia/Tokyo",
        "london": "Europe/London",
        "paris": "Europe/Paris",
        "sydney": "Australia/Sydney",
        "dubai": "Asia/Dubai"
    }
    city_key = city.lower().strip()
    tz = timezones.get(city_key)
    if not tz:
        return "I don't know the timezone for that city."
    city_time = datetime.now(pytz.timezone(tz))
    return f"The time in {city.title()} is {city_time.strftime('%H:%M:%S')}."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def get_bot_response():
    user_input = request.json["msg"]
    response = ""

    emotion, confidence = detect_emotion(user_input)
    if confidence > 0.6:
        if emotion == "anger":
            response += "I sense you're angry. Want to talk about it? 😟 "
        elif emotion == "joy":
            response += "That's wonderful! I'm happy for you! 😄 "
        elif emotion == "sadness":
            response += "I'm here for you. Sometimes talking helps. 💙 "
        elif emotion == "fear":
            response += "Fear is valid. I'm here if you need support. 🤗 "
        elif emotion == "surprise":
            response += "Whoa! Something unexpected happened? 🤯 "
        elif emotion == "love":
            response += "That's sweet! ❤️ "

    if "weather in" in user_input.lower():
        match = re.search(r"weather in ([a-zA-Z\s]+)", user_input.lower())
        if match:
            city = match.group(1).strip()
            response += get_weather(city)
    elif "time in" in user_input.lower() or "time is it in" in user_input.lower():
        match = re.search(r"time.*in ([a-zA-Z\s]+)", user_input.lower())
        if match:
            city = match.group(1).strip()
            response += get_time(city)
    else:
        fallback = chat.respond(user_input.lower())
        response += fallback if fallback else "I'm not sure how to respond to that."

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
