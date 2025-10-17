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
    text = ' '.join(text.split())

    return text

#get English stopwords
stop_words = set(stopwords.words('english'))

#removing stop words from text/words that dont add meaning
def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

#apply cleaning and stopword removal to the dataset
df['processed_review'] = df['review'].apply(clean_text).apply(remove_stopwords)

#convert sentiment labels to binary values
df['sentiment_label'] = df['sentiment'].map({'positive': 1, 'negative': 0})