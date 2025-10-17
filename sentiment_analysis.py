import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# load the IMDB dataset
df = pd.read_csv('IMDB Dataset.csv')

# download NLTK stopwords
nltk.download('stopwords')

# clean and preprocess text data
def clean_text(text):

    #remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    #remove special characters and numbers (only keep letters and spaces)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    #convert to lowercase
    text = text.lower()

    #remove extra whitespace
    text = " ".join(text.split())

    return text

