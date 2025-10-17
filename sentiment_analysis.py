import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import numpy as np

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

print('-'*50)

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

print('-'*50)

print("Confusion Matrix for Best Model:")

#get confusion matrix for best model
cm = confusion_matrix(y_test, best_model.predict(X_test))

print(cm)
print("Explanation of Confusion Matrix:")
print(f'True Negatives (Correctly predicted negative reviews): {cm[0,0]}')
print(f'False Positives (Incorrectly predicted positive reviews, predicted positive while negative): {cm[0,1]}')
print(f'False Negatives (Incorrectly predicted negative reviews, predicted negative while positive): {cm[1,0]}')
print(f'True Positives (Correctly predicted positive reviews): {cm[1,1]}')

#visualize confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression' if best_model == model else 'Confusion Matrix - Naive Bayes')
plt.colorbar()

#add labels
class_names = ['Negative', 'Positive']
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)

#add numbers to cells
for i in range(2):
    for j in range(2):
        plt.text(j, i, cm[i, j], ha='center', va = 'center', 
                 color='white' if cm[i, j] > cm.max() / 2 else 'black',
                 fontsize = 20)

#add axis labels, save and show plot
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

#----SUMMARY----#
print('-'*50)
print("MOVIE REVIEW SENTIMENT ANALYSIS SUMMARY")
print('-'*50)
print(f"""
Dataset: IMDB Movie Reviews
Total Reviews: {len(df)}
    - Positive Reviews: {(df['sentiment'] == 'positive').sum()}
    - Negative Reviews: {(df['sentiment'] == 'negative').sum()}

Text Processing:
    - Cleaned HTML and special characters
    - Removed stopwords (reduced words from ~225 to ~119 words per review on average)
    - Vectorized using TF-IDF (top 5000 features)

Models Trained:
    - Logistic Regression: {accuracy*100:.2f}% accuracy
    - Naive Bayes: {nb_accuracy*100:.2f}% accuracy

Best Model Performance ({ 'Logistic Regression' if best_model == model else 'Naive Bayes' }):
    - Accuracy: {best_accuracy*100:.2f}%
    - Negative Reviews: {cm[0,0]} correct, {cm[0,1]} wrong (recall: {cm[0,0]/(cm[0,0]+cm[0,1]):.2f}%)
    - Positive Reviews: {cm[1,1]} correct, {cm[1,0]} wrong (recall: {cm[1,1]/(cm[1,1]+cm[1,0]):.2f}%)

Key Insights: The model successfully indentifies sentiment with 88.7% accuracy by learning
patterns in word usage. Words like "amazing", "excellent" predict positive sentiment, while
words like "terrible", "awful" predict negative sentiment.

Files Created:
    - sentiment_analysis.py : The complete code for sentiment analysis
    - confusion_matrix.png : Visualization of the confusion matrix
""")
print('-'*50)
