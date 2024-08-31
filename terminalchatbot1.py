import random
import json
import pickle
import numpy as np
import nltk
import re
from datetime import datetime
import spacy
# import streamlit as st
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents3.json').read())

words = pickle.load(open('words3.pkl', 'rb'))
classes = pickle.load(open('classes3.pkl', 'rb'))
model = load_model('test3.h5')
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)
def predict_class (sentence):
    bow = bag_of_words (sentence)
    
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    # print(res)
    # result1=[[i, r] for i, r in enumerate(res)]
    # result1.sort(key=lambda x: x[1],reverse=True)
    # print(result1)
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

def main():
    while(True):
        message = input("Enter messagae :")

        ints = predict_class (message)
        print(ints)
        # for i in range(len(ints)):
        flag=0
        for i in range(len(ints)):
            if(float(ints[i]["probability"])>=0.5):
                if(ints[i]['intent']=='book_ticket' or ints[i]['intent']=='parchi_kaatna'):
                    user_text=message
                    date_pattern = (
                        r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b|'
                        r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b|'
                        r'\b(\d{1,2})(?:st|nd|rd|th)? (\w{3,9})(?: (\d{4}))?\b|'
                        r'\b(\d{1,2}) (\w{3,9})(?:,? (\d{4}))?\b'
                    )

                    city_pattern = r'\b(?:for|in)\s+([\w\s]+)$'

                    def validate_date(day, month, year):
                        try:
                            date = datetime(year, month, day)
                            return date
                        except ValueError:
                            return None

                    def month_name_to_number(month_name):
                        month_name = month_name.lower()
                        month_map = {
                            'jan': 1, 'january': 1,
                            'feb': 2, 'february': 2,
                            'mar': 3, 'march': 3,
                            'apr': 4, 'april': 4,
                            'may': 5, 'may': 5,
                            'jun': 6, 'june': 6,
                            'jul': 7, 'july': 7,
                            'aug': 8, 'august': 8,
                            'sep': 9, 'september': 9,
                            'oct': 10, 'october': 10,
                            'nov': 11, 'november': 11,
                            'dec': 12, 'december': 12
                        }
                        return month_map.get(month_name, None)

                    date_match = re.search(date_pattern, user_text)
                    date = None
                    current_year = datetime.now().year

                    if date_match:
                        if date_match.group(1):
                            day, month, year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
                            if year < 100:
                                year += 2000
                            date = validate_date(day, month, year)
                        elif date_match.group(4):
                            day, month, year = int(date_match.group(4)), int(date_match.group(5)), int(date_match.group(6))
                            if year < 100:
                                year += 2000
                            date = validate_date(day, month, year)
                        elif date_match.group(7):
                            day = int(date_match.group(7))
                            month = month_name_to_number(date_match.group(8).lower())
                            year = int(date_match.group(9)) if date_match.group(9) else current_year
                            date = validate_date(day, month, year)
                        elif date_match.group(10):
                            day = int(date_match.group(11))
                            month = month_name_to_number(date_match.group(10).lower())
                            year = int(date_match.group(12)) if date_match.group(12) else current_year
                            date = validate_date(day, month, year)

                    nlp = spacy.load("en_core_web_sm")

                    museum_list = {
                        "Delhi Science Center": "Delhi%20Science%20Museum",
                        "Delhi Science Museum": "Delhi%20Science%20Museum",
                    }

                    def extract_museum_from_list(text, museums):
                        text = text.lower()
                        for museum in museums:
                            if museum.lower() in text:
                                return museums[museum]
                        return None

                    museum = extract_museum_from_list(user_text, museum_list)

                    if not museum:
                        museum = 'Not found'

                    print(f"Museum: {museum if museum else 'Not found'}")
                    print(f"Date: {date.strftime('%d-%m-%Y') if date else 'Invalid date'}")
                flag=1
                res = get_response (ints, intents)
                print(res)
                break
        if(flag==0):
            print("For this query I don't know the answer please contact (033) 23576008.")
        
        if "bye" in message.lower():
            break
            

if __name__ == "__main__":
    main()