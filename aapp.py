import streamlit as st
import pickle
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already downloaded
nltk.download('stopwords')


# Function to transform text by removing stopwords and converting to lowercase
def text_transform(text):
    new_list = []
    text = text.lower()
    words = text.split()
    for i in words:
        if i not in stopwords.words('english'):
            new_list.append(i)
    return " ".join(new_list)


# Load the vectorizer and model
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

# Title of the Streamlit app
st.title("Email Spam Classifier")

# Input from the user
input_mail = st.text_area("Enter the message")

# On button click, classify the input message
if st.button('Check'):
    transform_email = text_transform(input_mail)
    vector_input = tfidf.transform([transform_email])

    # Make prediction using the model
    result = model.predict(vector_input)[0]

    # Display the result
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")
