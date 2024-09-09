import random
import json
import pickle
import numpy as np
import nltk
import re
from datetime import datetime,timedelta
import spacy
# import streamlit as st
import os
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from fuzzywuzzy import process
from flask import Flask, request, jsonify

app = Flask(__name__)

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents4.json').read())
words = pickle.load(open('ticket_bot_words.pkl', 'rb'))
classes = pickle.load(open('ticket_bot_classes.pkl', 'rb'))
model = load_model('ticket_bot.h5')

intents_booking=json.loads(open('booking_intent.json').read())
words_booking=pickle.load(open('booking_words.pkl','rb'))
classes_booking=pickle.load(open('booking_classes.pkl','rb'))
model_booking=load_model('booking_bot1.h5')

museum_list = {
    "Delhi Science Center": "National Science Centre Delhi",
    "Delhi Science Museum": "National Science Centre Delhi",
    "National science museum":'National Science Centre Delhi',
    "National science center":'National Science Centre Delhi',
    "Delhi museum":'National Science Centre Delhi'
    }


def extract_museum_from_list(text, museums):
    text = text.lower()
    for museum in museums:
        if museum.lower() in text:
            return museums[museum]
    return None

def validate_date1(date_str):
    try:
        # Parse the input date string in the dd/mm/yyyy format
        input_date = datetime.strptime(date_str, '%d-%m-%Y')
        # Get today's date without the time part
        today = datetime.today().date()
        # Check if the input date is in the past
        if input_date.date() < today:
            return False  # Invalid if it's older than today's date
        return True  # Valid if the date is today or in the future
    except ValueError:
        return False  # Invalid if the format doesn't match

def validate_date(day, month, year):
    try:
        if None in (day, month, year):
            return None
        date = datetime(year, month, day)
        if date < datetime.now().replace(hour=0, minute=0, second=0, microsecond=0):
            return None
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

def extract_date_from_keywords(text):
    if re.search(r'\bphele\b|\bbefore\b', text, re.IGNORECASE):
        return None
    if re.search(r'\baaj se\b|\bfrom today\b', text, re.IGNORECASE):
        match = re.search(r'(\d+)\s*(?:days)?\s*(?:aaj se|from today)', text, re.IGNORECASE)
        if not match:
            match = re.search(r'(?:aaj se|from today)\s*(\d+)\s*(?:days)?', text, re.IGNORECASE)
        if match:
            days_to_add = int(match.group(1))
            return datetime.now() + timedelta(days=days_to_add)
    elif re.search(r'\btoday\b|\baaj\b', text, re.IGNORECASE):
        return datetime.now()
    elif re.search(r'\bday after tomorrow\b|\bperso\b', text, re.IGNORECASE):
        return datetime.now() + timedelta(days=2)
    elif re.search(r'\btomorrow\b|\bkal\b', text, re.IGNORECASE):
        return datetime.now() + timedelta(days=1)
    return None

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

def bag_of_words_booking (sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words_booking)
    for w in sentence_words:
        for i, word in enumerate(words_booking):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class (sentence):
    bow = bag_of_words (sentence)
    
    res = model.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({'intent': classes [r[0]], 'probability': str(r[1])})
    return return_list


def predict_class_booking (sentence):
    bow = bag_of_words_booking (sentence)
    
    res = model_booking.predict(np.array([bow]))[0]
    
    ERROR_THRESHOLD = 0.70
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    # print(results)
    return_list = []
    for r in results:
        return_list.append({'intent': classes_booking [r[0]], 'probability': str(r[1])})
    return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    result=""
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice (i['responses'])
            break
    return result

def validate(d):
    for i in d:
        if(d[i]==None):
            return [False,i]
    return [True]



# def main():
#     while(True):
#         message = input("Enter messagae :")

#         ints = predict_class (message)
#         print(ints)
#         # for i in range(len(ints)):
#         flag=0
#         for i in range(len(ints)):
#             if(float(ints[i]["probability"])>=0.5):
#                 if(ints[i]['intent']=='book_ticket' or ints[i]['intent']=='parchi_kaatna'):
#                     user_text=message
#                     date_pattern = (
#                         r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b|'
#                         r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b|'
#                         r'\b(\d{1,2})(?:st|nd|rd|th)? (\w{3,9})(?: (\d{4}))?\b|'
#                         r'\b(\d{1,2}) (\w{3,9})(?:,? (\d{4}))?\b'
#                     )

#                     # city_pattern = r'\b(?:for|in)\s+([\w\s]+)$'

#                     def validate_date(day, month, year):
#                         try:
#                             date = datetime(year, month, day)
#                             return date
#                         except ValueError:
#                             return None

#                     def month_name_to_number(month_name):
#                         month_name = month_name.lower()
#                         month_map = {
#                             'jan': 1, 'january': 1,
#                             'feb': 2, 'february': 2,
#                             'mar': 3, 'march': 3,
#                             'apr': 4, 'april': 4,
#                             'may': 5, 'may': 5,
#                             'jun': 6, 'june': 6,
#                             'jul': 7, 'july': 7,
#                             'aug': 8, 'august': 8,
#                             'sep': 9, 'september': 9,
#                             'oct': 10, 'october': 10,
#                             'nov': 11, 'november': 11,
#                             'dec': 12, 'december': 12
#                         }
#                         return month_map.get(month_name, None)

#                     date_match = re.search(date_pattern, user_text)
#                     date = None
#                     current_year = datetime.now().year

#                     if date_match:
#                         if date_match.group(1):
#                             day, month, year = int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3))
#                             if year < 100:
#                                 year += 2000
#                             date = validate_date(day, month, year)
#                         elif date_match.group(4):
#                             day, month, year = int(date_match.group(4)), int(date_match.group(5)), int(date_match.group(6))
#                             if year < 100:
#                                 year += 2000
#                             date = validate_date(day, month, year)
#                         elif date_match.group(7):
#                             day = int(date_match.group(7))
#                             month = month_name_to_number(date_match.group(8).lower())
#                             year = int(date_match.group(9)) if date_match.group(9) else current_year
#                             date = validate_date(day, month, year)
#                         elif date_match.group(10):
#                             day = int(date_match.group(11))
#                             month = month_name_to_number(date_match.group(10).lower())
#                             year = int(date_match.group(12)) if date_match.group(12) else current_year
#                             date = validate_date(day, month, year)

#                     nlp = spacy.load("en_core_web_sm")

#                     museum_list = {
#                         "Delhi Science Center": "Delhi%20Science%20Museum",
#                         "Delhi Science Museum": "Delhi%20Science%20Museum",
#                     }

#                     def extract_museum_from_list(text, museums):
#                         text = text.lower()
#                         for museum in museums:
#                             if museum.lower() in text:
#                                 return museums[museum]
#                         return None

#                     museum = extract_museum_from_list(user_text, museum_list)

#                     if not museum:
#                         museum = 'Not found'

#                     print(f"Museum: {museum if museum else 'Not found'}")
#                     print(f"Date: {date.strftime('%d-%m-%Y') if date else 'Invalid date'}")
#                 flag=1
#                 res = get_response (ints, intents)
#                 print(res)
#                 break
#         if(flag==0):
#             print("For this query I don't know the answer please contact (033) 23576008.")
        
#         if "bye" in message.lower():
#             break
booking_flag=0
name_flag=0
location_flag=0
date_flag=0
# update_flag=0
count=0
d={"Museum_location":None,"visit_date":None,"ticket_type":None}
print(count)
@app.route('/process', methods=['POST'])
def process_string():
    global booking_flag, name_flag, location_flag, date_flag, count
    global d
    ticketPrices = {
        'General Entry': 70,
        'General Entry (Group >25)': 60,
        'General Entry (BPL Card)': 20,
        'Students Entry (School Group)': 25,
        'Students Entry (Govt/MCD School)': 10,
        '3D Film': 40,
        'SDL/Taramandal (Adult)': 20,
        'SDL/Taramandal (Children)': 20,
        'SOS Entry (Adult)': 50,
        'Holoshow Entry (Adult)': 40,
        'Fantasy Ride': 80,
        'Package (All Inclusive)':250
    }
    museum_list=[
        "National Science Centre Delhi",
        "National Railway Museum",
        "Victoria Memorial Hall"
        "Allahabad Museum",
        "Salar Jung Museum"
    ]
    
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the string from the JSON
    message = data.get('input_string')
    
    res_flag=0

    ints = ""
    if(booking_flag==0):
        ints= predict_class(message)
    else:
        ints=predict_class_booking(message)
    print(booking_flag)
    print(ints)
    if(len(ints)!=0):
        if( ints[0]['intent']=='book_ticket'):
            booking_flag=1
            print()
            print(museum_list)
            print()
            print()
            print(ticketPrices)
            print()
            response_data={
                "output": "\n".join(museum_list) + "\n" + str(ticketPrices)
            }
            return jsonify(response_data), 200
        
        if(booking_flag==1 and validate(d)[0]==False):
            if(booking_flag==1): count+=1
            if(booking_flag==1 and validate(d)[0]==False and count>1):
                if(ints[0]['intent'] not in ("find_location","find_type","find_date")):
                    response_data={
                        "output": f"I want to know your {validate(d)[1]} inorder to book ticket"
                    }
                    return jsonify(response_data),200
                else:
                    if(ints[0]['intent']=='find_location'): # checking location
                        
                        
                        user_text=message.lower()
                        museum_patterns = {
                            "National Science Centre Delhi": r"\b(science|national\s+science)\b",
                            "National Railway Museum": r"\b(railway|national\s+railway)\b",
                            "Victoria Memorial Hall": r"\b(victoria\s+memorial|memorial\s+hall)\b",
                            "Allahabad Museum": r"\b(allahabad\s+museum)\b",
                            "Salar Jung Museum": r"\b(salar\s+jung|jung\s+museum)\b"
                        }
                        best_match = None
                        for museum, pattern in museum_patterns.items():
                            if re.search(pattern, user_text):
                                best_match = museum
                                break
                                
                        if best_match:
                            d['Museum_location']=best_match
                        else:
                            
                            print("Didn't Get museum name")
                            d['Museum_location']=None
                            
                    elif(ints[0]['intent']=='find_type'): #checking ticket type
                        
                        user_text=message
                        keywords = {
                            "sos": "SOS Entry (Adult)",
                            "general entry": "General Entry",
                            "group more than 25": "General Entry (Group >25)",
                            "bpl": "General Entry (BPL Card)",
                            "school group": "Students Entry (School Group)",
                            "govt school": "Students Entry (Govt/MCD School)",
                            "mcd school": "Students Entry (Govt/MCD School)",
                            "3d film": "3D Film",
                            "taramandal": "SDL/Taramandal",
                            "fantasy ride": "Fantasy Ride",
                            "holoshow": "Holoshow Entry (Adult)",
                            "package": "Package (All Inclusive)"
                        }
                        
                        matched_ticket = None
                        for keyword, ticket in keywords.items():
                            if keyword in user_text.lower():
                                matched_ticket = ticket
                                break
                        
                        if not matched_ticket:
                            ticket_names = list(keywords.keys())
                            best_match, score = process.extractOne(user_text, ticket_names)
                            if score > 50:
                                matched_ticket = best_match
                            
                        if matched_ticket:
                            d['ticket_type']=matched_ticket
                        else:
                            d['ticket_type']=None
                            
                    elif(ints[0]['intent']=='find_date'): # checking date
                        
                        
                        user_text=message
                        date_pattern = (
                            r'\b(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})\b|'
                            r'\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b|'
                            r'\b(\d{1,2})(?:st|nd|rd|th)? (\w{3,9})(?: (\d{4}))?\b|'
                            r'\b(\d{1,2}) (\w{3,9})(?:,? (\d{4}))?\b'
                        )
                        date = extract_date_from_keywords(user_text)
                        if not date:
                            date_match = re.search(date_pattern, user_text)
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
                        if date:
                            try:
                                d['visit_date']=date.strftime('%d-%m-%Y')
                            except AttributeError:
                                print("Invalid date")
                # print(validate(d)[0])
        
        elif(booking_flag==1 and validate(d)[0]==True):
            booking_flag=0
            response_data={
                "output":"All info about your ticket is collected"
            }
            return jsonify(response_data),200
            # print(d)
        if(booking_flag==1 and validate(d)[0]==True ):
            # print(validate(d)[0])
            
            res = get_response(ints, intents_booking)
            print(res)
            print()
            print(d)
            # d={"Museum_location":None,"visit_date":None,"ticket_type":None}
            response_data={
                "output":res,
                "ticket":{
                    "museum_location":d['Museum_location'],
                    "visit_date":d['visit_date'],
                    "ticket_type":d['ticket_type']
                    
                }
            }
            booking_flag=0
            res_flag=1
            return jsonify(response_data),200
            
        if(booking_flag==1 and res_flag==0 and count>1):
            res = get_response(ints, intents_booking)
            response_data={
                "output":res
            }
            print(res)
            return jsonify(response_data),200
        elif(booking_flag==1 and res_flag==0 and count<2):
            res= get_response(ints,intents)
            print(res)
            response_data={
                "output":res
            }
            return jsonify(response_data),200
            
        if(booking_flag==0 and res_flag==0 ):
            res= get_response(ints,intents)
            print(res)
            response_data={
                "output":res
            }
            return jsonify(response_data),200
    else:
        response_data={
                "output":"I did not get what you said"
            }
        return jsonify(response_data),200
    if "bye" in message.lower():
        response_data={
                "output":""
            }
        return jsonify(response_data),200

            

if __name__ == '__main__':
    app.run(debug=True)