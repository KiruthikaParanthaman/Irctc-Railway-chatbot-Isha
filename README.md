# Irctc-Railway-Chatbot-Isha
Meet 'Isha" :  Sibling of Irctc chatbot Disha.2.0. Disha is too formal, while Isha is Jovial. What can Isha do? She can give you trains between two stations as well have a small coffee-chat!!!

**Who is Isha?**:
 - Isha is chatbot inspired from Disha 2.0 Irctc official chatbot
 - Isha is developed for experimental and personal learning purposes
 - Isha is trained using small set of publicy available frequently asked questions(faq's) in irctc official website,general conersations corpus from chatterbot and personalized question sets like what is your name?
 - Isha is **hybrid model** - using both Rule based approach and Machine Learning approach
 - **Chatterbot** library is used for training the chatbot with customised training dataset
 - Rule based approach is used when user prompts for booking tickets
 - Chatbot guides the user with prompts about source and destination station codes and helps in re-entering details if error occurs
 - Isha fetches irctc train details  based on publicly available upi resources. There could be mismatches between the official train lists. As it is experiemntal, the accuracy of informations is not guaranteed.
 - Isha replies with default response if best matches couldnt be found in the training dataset
 - User prompts are pre-processed with methods like **lowercasing,Tokenizing,spelling correction,Part of Speech Tagging,White Space removal,Lemmatization,spelling correction** 


**What can Isha do?**:
- Though developed for booking tickets, at present Isha will give only **train details between two stations** when prompted with book tickets
- Isha can **handle small conversations** like greetings,general questions related to ticket booking like can i cancel confirmed tickets?
- 

**How it is different from Disha 2.0**:
- Disha is Rule based algorithm, i.e., it can handle requests only when prompted with exact questions like for booking user should type "I want to book ticket".If user prompts otherwise like could you book ticket or book ticket Disha Disha replies with default apology response. But Isha has little bit of flexible approach.It checks for best matches with trained dataset
- Disha cannot handle small greetings like "hi","hello".Isha is equipped to handle such greetings

**Preview of application**:
![isha chat1](https://github.com/KiruthikaParanthaman/Irctc-Railway-chatbot-Isha/assets/141828622/463eced1-0fd9-4f6b-9b35-c6ee6483ee76)


