import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

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

#vectorize text data using TF-IDF using only top 5000 most important words
vectorizer = TfidfVectorizer(max_features=5000)

#convert all reviews to numbers
X = vectorizer.fit_transform(df['processed_review'])
y = df['sentiment_label']

#split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   


#create and train model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

#make predictions on test set
y_pred = model.predict(X_test)

#evaluate the accuracy of model
accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Model")
print('-'*50)
print(f'Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)')

print('\nClassification Report:')
print(classification_report(y_test, y_pred, target_names=['negative', 'positive']))