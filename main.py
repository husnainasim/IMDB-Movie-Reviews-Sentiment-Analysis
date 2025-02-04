import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Dense,SimpleRNN
from tensorflow.keras.models import load_model
import streamlit as st



#Load the Word index
word_index=imdb.get_word_index()
reversed_word_index={value: key for key,value in word_index.items()}

#Load the Pre-trained model 
model=load_model('simple_rnn_imdb.h5')


#Step 2 Helper Functions
#Fucntion to decode the reviews

def decode_review(encoded_review):
    decoded_review=' '.join([reversed_word_index.get(i-3,'?') for i in encoded_review])
    return decoded_review

#Function to preprocess the input review(text-->encoded_review(in the format of decimal values or you can say binary)-->padded_review)
def preprocess_text(review):
    words=review.lower().split()
    encoded_review=[word_index.get(word,2) for word in words]
    padded_review=sequence.pad_sequences([encoded_review],maxlen=500)
    return padded_review

#Prediction Function
def predict_sentiment(review):
    preprocessed_input=preprocess_text(review)
    
    prediction=model.predict(preprocessed_input)
    
    sentiment='Positive' if prediction[0][0]>0.5 else 'Negative'
    
    return sentiment,prediction[0][0]

#? Streamlit Web Appp

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify as positive or negative.')

#User Input

user_input=st.text_area('Movie Review:')

if st.button('Classify'):
    
    sentiment,score=predict_sentiment(user_input)
    
    #Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {score}')
else:
    st.write('Please enter a movie review!!')

