#Import the necessary packages
import streamlit as st
import random
import time
import json
from urllib.request import urlopen
from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
from chatterbot.trainers import ListTrainer
import nltk
import string
from textblob import TextBlob 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import spacy
import collections.abc
collections.Hashable = collections.abc.Hashable
spacy.load("en_core_web_sm")

#App Title
st.title("Isha")

#Function to pre-process user input    
def preprocess(sentence,return_type):    
    #Removing whitespaces
    sentence = sentence.strip()
    sentence = " ".join(sentence.split())
    #Correcting spelling mistakes
    spell_proof = str(TextBlob(sentence).correct())
    # tokenize the text
    word_tokens = nltk.word_tokenize(spell_proof)
    #Lowercasing words
    lowercase_words = [word.lower() for word in word_tokens]
    #Removing Punctuation
    punc_less_words = [word for word in lowercase_words if word not in string.punctuation]
    if return_type == 'words':
        return punc_less_words
    elif return_type == 'full_sentence':
        return (" ".join(word for word in punc_less_words))       
        

#Function for part of speech tagging
def pos_tagger(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:          
        return None    

#Function to lemmatize with pos tagging
# converts word 'booking' to 'book' after pos tagging. Lemmatizing without pos tagging returns word 'booking' as such
def lemmatizer(sentence):
    sentence = preprocess(sentence,'words')  
    #pos-tagging
    #Code inspired from geeksforgeeks
    pos_tagged = nltk.pos_tag(sentence)
    wordnet_tagged = list(map(lambda x: (x[0], pos_tagger(x[1])),pos_tagged))
    #Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_words= []
    for word, tag in wordnet_tagged:
        if tag is None:
            # if there is no available tag, append the token as is
            lemmatized_words.append(word)
        else:        
            # else use the tag to lemmatize the token
            lemmatized_words.append(lemmatizer.lemmatize(word, tag))
    #Remove stop words
    stopwords = nltk.corpus.stopwords.words("english")
    filtered_words = [word for word in lemmatized_words if word not in stopwords]
    return filtered_words

#Function to fetch train details using upi
#Input source and destination station code in caps.Returns dataframe
def train_details(source,destination):
    url = f"https://indian-railway-api.cyclic.app/trains/betweenStations/?from={source}&to={destination}"
    # Fetching the URL 
    response = urlopen(url)   
    # Loading the fetched json data 
    data_json = json.loads(response.read()) 
    train_num = []
    train_name = []
    departure = []
    arrival = []
    travel_time = []
    try:
        if data_json:
            for train in data_json['data']:
                train_num.append(train['train_base']['train_no'])
                train_name.append(train['train_base']['train_name'])
                departure.append(train['train_base']['from_time'])
                arrival.append(train['train_base']['to_time'])
                travel_time.append(train['train_base']['travel_time'])    
            df = pd.DataFrame(list(zip(train_num, train_name,departure,arrival,travel_time)),
               columns =['Train Number', 'Train name','Departure Time','Arrival Time','Travel Time'])
            return df
        else:
            return False
    except:
        return False

#Chatterbot chatbot trained using irctc faq yml file
chatbot = ChatBot('Isha',storage_adapter='chatterbot.storage.SQLStorageAdapter',
                  logic_adapters = [{
        'import_path': 'chatterbot.logic.BestMatch',
        'default_response': 'I am sorry, I cannot help you with that as i do not have necessary information.Try typing book ticket',
                }],
                database_uri='sqlite:///database.db',
                read_only=True)
trainer1 = ChatterBotCorpusTrainer(chatbot)
trainer1.train("https://github.com//KiruthikaParanthaman//Irctc-Railway-chatbot-Isha//blob/main//irctc_train_data.yml")


#function for streamlit write stream display
def display(number):
    response = {1 : "Sure.From which station do you want the train?Enter source Station code",
                2 : "Till which station do you want the train?",
                3 : "Sorry.Entered source or destination station code is wrong.Do you want to try again? Type Yes or no",
                4 : "Enter source Station code again",
                5 : "Thank you for using the service"}
    for word in response[number].split():
        yield word + " "
        time.sleep(0.05)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
#Initialize status
if "status" not in st.session_state:
    st.session_state.status = []
#Initialize Source
if "source" not in st.session_state:
    st.session_state.source = []
#Initailize Destination
if "destination" not in st.session_state:
    st.session_state.destination = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        try:
            st.dataframe(message['content'][1])
        except:
            st.markdown(message['content'])

#Get user chat input and compare processed texts to find words book and ticket if any
#Function to get source code and destination code and display information.Option to retry and exit 
prompt = st.chat_input("Hi,I'm Isha") 
if prompt:
    #pre-Process user input 
    preprocessed = lemmatizer(prompt)
    words = ['book','ticket']
    decision = ["yes",'no']

    #If matching words in 'words' with 'preprocessed' words enter if condition and Prompt for station source code
    if (set(words).issubset(preprocessed)) and ("cancel" not in preprocessed):
        # Display user message in chat message container and append to messages        
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            response = st.write(prompt)
        with st.chat_message("Isha"):
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "Isha", "content": "Sure.From which station do you want the train?Enter source Station code"})
            #Enable status by appending 1:
            st.session_state.status.append(1)
            response = st.write_stream(display(1)) 

    #Save Source code and prompt about Destination code 
    elif (len(st.session_state.status)!=0) and(len(st.session_state.source)==0) and (len(st.session_state.destination)==0):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            response = st.write(prompt)
        st.session_state.source.append(prompt.upper())
        with st.chat_message("Isha"):
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "Isha", "content": "Till which station do you want the train?"})
            response = st.write_stream(display(2))
      
    #Display train details in dataframe format and initialize status,source,destination as empty  
    elif(len(st.session_state.status)!=0) and(len(st.session_state.source)!=0) and (len(st.session_state.destination)==0):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            response = st.write(prompt)
        st.session_state.destination.append(prompt.upper())
        train_info = train_details(st.session_state.source[0],st.session_state.destination[0])
        if type(train_info)!= bool:
            with st.chat_message("Isha"):
                st.session_state.messages.append({"role": "Isha", "content": {1 : train_info}})
            st.dataframe(train_info)
            st.session_state.status = []
            st.session_state.source = []
            st.session_state.destination = []  

        #If Error prompt whether customer wants to retry              
        else:
            with st.chat_message("Isha"):
                st.session_state.messages.append({"role": "Isha", "content": "Sorry.Entered source or destination station code is wrong.Do you want to try again? Type Yes or no"})
                response = st.write_stream(display(3))

    #If cutomer enters "yes" prompt with source code and continue the loop             
    elif  (len(st.session_state.source)!=0) and (len(st.session_state.destination)!=0) and ((prompt.lower()) == "yes"):
        print(preprocessed)
        if "yes" in preprocessed:
            print("inside yes")
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                response = st.write(prompt)
            with st.chat_message("Isha"):
            # Add assistant response to chat history
                st.session_state.messages.append({"role": "Isha", "content": "Enter source Station code again"})
                st.session_state.source = []
                st.session_state.destination = [] 
                #Enable status :
                st.write_stream(display(4))  
                             
    #if customer enters 'no' exit the condition.Re-initialize source,destination and status empty
    elif  (len(st.session_state.source)!=0) and (len(st.session_state.destination)!=0) and ((prompt.lower()) == "no"):
        print("insode no")
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            print("inside st.chat.user")
            response = st.write(prompt)                        
        with st.chat_message("Isha"):
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "Isha", "content": "Thank you for using the service"})
            st.session_state.source = []
            st.session_state.destination = []
            st.session_state.status = []
            #Write into chat input :
            st.write_stream(display(5))
                
    #Display the best matching response based on the chatbot trained data                 
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            response = st.write(prompt)
        with st.chat_message("Isha"):
            bot_response =chatbot.get_response(preprocess(prompt,'full_sentence'))
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "Isha", "content": bot_response})
            response = st.write(bot_response)





