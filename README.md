# Sentiment Analysis on IMDB Movie Reviews

A Natural Language Processing (NLP) project that predicts whether a movie review is positive or negative using machine learning classification algorithms.

## 📊 Project Overview

This project analyzes 50,000 IMDB movie reviews to build a sentiment classifier. The model processes raw text, extracts meaningful features, and predicts sentiment with high accuracy.

- **Dataset**: IMDB Dataset of 50K Movie Reviews from Kaggle
- **Total Reviews**: 50,000 (25,000 positive, 25,000 negative)
- **Features**: 5,000 most important words extracted using TF-IDF
- **Target Variable**: Sentiment (Positive/Negative)

## 🎯 Objective

Build an NLP model that can automatically determine if a movie review expresses positive or negative sentiment based on the text content.

## 🛠️ Technologies Used

- **Python 3.13.5**
- **pandas** - Data manipulation and analysis
- **numpy** - Numerical operations and array handling
- **scikit-learn** - Machine learning algorithms and text vectorization
- **nltk** - Natural language processing and text preprocessing
- **matplotlib** - Data visualization
- **re** - Regular expressions for text cleaning

## 📁 Project Structure
```
sentiment_analysis/
├── sentiment_analysis.py      
├── IMDB Dataset.csv          
├── confusion_matrix.png       
├── requirements.txt           
└── README.md                  
```

## 🔍 Process

### 1. Text Preprocessing
- Removed HTML tags and special characters
- Converted all text to lowercase
- Removed punctuation and numbers
- Eliminated stopwords (common words like "the", "is", "and")
- Reduced average review length from ~225 to ~119 meaningful words

### 2. Feature Engineering
- Applied TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
- Extracted 5,000 most important features from the vocabulary
- Converted text data into numerical vectors for machine learning

### 3. Model Training
Trained and compared two classification models:

- **Logistic Regression**: 88.74% accuracy
- **Naive Bayes**: 84.99% accuracy

### 4. Model Evaluation
Used confusion matrix and classification metrics to analyze performance:
```
Confusion Matrix (Logistic Regression):
- True Negatives: 4,316 (correctly identified negative reviews)
- False Positives: 645 (incorrectly predicted as positive)
- False Negatives: 480 (incorrectly predicted as negative)
- True Positives: 4,559 (correctly identified positive reviews)
```

## 📈 Results

- **Best Model**: Logistic Regression
- **Accuracy**: 88.74%
- **Negative Review Recall**: 87% (caught 4,316 out of 4,961)
- **Positive Review Recall**: 90% (caught 4,559 out of 5,039)
- **Balanced Performance**: Model performs equally well on both classes

## 💡 Key Findings

### Top Predictive Patterns:
- **Positive indicators**: Words like "excellent", "amazing", "brilliant", "loved", "perfect"
- **Negative indicators**: Words like "terrible", "awful", "waste", "worst", "boring"
- **Word importance**: TF-IDF successfully identified sentiment-bearing words while filtering out common terms

### Technical Insights:
- Stopword removal reduced vocabulary by ~47% while maintaining meaning
- TF-IDF with 5,000 features captured sufficient information for high accuracy
- Logistic Regression outperformed Naive Bayes by 3.75% on this dataset

## 🚀 How to Run

1. Clone this repository
```bash
git clone https://github.com/MatthewMo520/movie-sentiment-analysis
cd sentiment_analysis
```

2. Install required packages
```bash
pip install -r requirements.txt
```

3. Download NLTK stopwords (first time only)
```python
import nltk
nltk.download('stopwords')
```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) and place it in the project folder

5. Run the analysis
```bash
python sentiment_analysis.py
```

## 📊 Sample Predictions

**Example 1:**
- Review: "This movie was absolutely amazing! Best film I've ever seen!"
- Prediction: ✅ Positive (Confidence: 95%)

**Example 2:**
- Review: "Terrible waste of time. Boring and poorly acted."
- Prediction: ✅ Negative (Confidence: 92%)

## 🔮 Future Improvements

- Implement deep learning models (LSTM, BERT) for potentially higher accuracy
- Add sentiment intensity (not just positive/negative, but how positive/negative)
- Build a web interface for real-time sentiment prediction
- Expand to multi-class classification (positive, neutral, negative)
- Apply the model to other domains (product reviews, tweets, customer feedback)
- Experiment with word embeddings (Word2Vec, GloVe) instead of TF-IDF

## 📚 What I Learned

- Text preprocessing techniques for NLP (tokenization, stopword removal, cleaning)
- TF-IDF vectorization and how to convert text to numerical features
- Difference between bag-of-words and weighted word importance
- Comparing multiple classification algorithms for text data
- Interpreting confusion matrices and classification reports for NLP tasks
- Handling large text datasets efficiently

## 👤 Author

Matthew Mo