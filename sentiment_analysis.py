import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.naive_bayes import MultinomialNB

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

#----logistic regression model----#

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
print('-'*50)

#----Naive Bayes model----#

#create and train model
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

#Predict on test set
y_nb_pred = nb_model.predict(X_test)

#evaluate the accuracy of model
nb_accuracy = accuracy_score(y_test, y_nb_pred)
print("Naive Bayes Model")
print('-'*50)
print(f'Accuracy: {nb_accuracy:.4f} ({nb_accuracy*100:.2f}%)')

print('\nClassification Report:')
print(classification_report(y_test, y_nb_pred, target_names=['negative', 'positive']))
print('-'*50)

#compare models
print("Model Comparison")
print('-'*50)
print(f'Logistic Regression: {accuracy*100:.2f}%')
print(f'Naive Bayes: {nb_accuracy*100:.2f}%')

#determine the better model
if nb_accuracy > accuracy:
    print(f'The better model is Naive Bayes (+{(nb_accuracy - accuracy)*100:.2f}%)')
    best_model = nb_model
    best_accuracy = nb_accuracy
else:
    print(f'The better model is Logistic Regression (+{(accuracy - nb_accuracy)*100:.2f}%)')
    best_model = model
    best_accuracy = accuracy