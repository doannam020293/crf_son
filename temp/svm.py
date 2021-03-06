# import numpy
from  pyvi import ViTokenizer
import pandas as pd
from sklearn import svm, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from unidecode import unidecode
import re

tfidf_vectorizer = TfidfVectorizer()
count_vectorizer = CountVectorizer()

def train_svm(data):
  X = extract_features(row[0] for row in data)
  x = [row[1] for row in data]
  # clf = svm.LinearSVC()
  clf = svm.SVC(kernel='linear', probability=True)
  clf.fit(X, x)
  return clf

def extract_features(docs):
  docs = map(preprocess_text, docs)
  features = count_vectorizer.fit_transform(docs)
  return features

def preprocess_text(text):
  text = ViTokenizer.tokenize(text)
  return text

# def remove_rt(text):
#   return re.sub('RT ', '', text)
#
# def remove_twitter_user_mentions(text):
#   return re.sub(r'(?:@[\w_]+)', '', text)
#
# def remove_hashtags(text):
#   return re.sub(r'(?:\#+[\w_]+[\w\'_\-]*[\w_]+)', '', text)
#
# def remove_links(text):
#   return re.sub(r'http\S+', '', text)
#
# def remove_special_chars(text):
#   return re.sub(r'[^\w\s\']', '', text)
#
# def remove_numbers(text):
#   return re.sub(r'\d', '', text)

def build_classification_report(clf, test_data):
  y_true = [row[1] for row in test_data]
  docs = map(preprocess_text, [row[0] for row in test_data])
  tfidf = count_vectorizer.transform(docs)
  y_pred = clf.predict(tfidf)
  report = classification_report(y_true, y_pred)
  return report

def cross_validation_report(clf, dataset):
  data = count_vectorizer.transform([row[0] for row in dataset])
  target = [row[1] for row in dataset]
  return cross_validation.cross_val_score(clf, data, target)

data = pd.read_csv('data.csv', encoding='utf8').as_matrix()
clf = train_svm(data)

if __name__ == '__main__':
  # numpy.set_printoptions(threshold='nan')
  train_data, test_data = train_test_split(data, test_size=0.2)
  print 'Training SVM...'
  clf = train_svm(train_data)
  print 'SVM trained'

  print 'Building reports...'
  print 'Classification report:'
  print build_classification_report(clf, test_data)
  print '----------'
  print 'Cross-validation report:'
  print cross_validation_report(clf, data)