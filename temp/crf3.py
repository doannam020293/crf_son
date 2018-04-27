# import string
# import os
# import sys
# import pickle
# from collections import Counter
# from  pyvi import ViTokenizer

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import pandas as pd


from sklearn import svm, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing



input_dir =  r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\data_input\nlp\chat bot\classification\data'


def train_save_model(df,name_save):
    # df = pd.read_excel(input_dir + r"\domain_full_file.xlsx")
    # df = pd.read_excel(input_file)
    count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    # le = preprocessing.LabelEncoder()

    X = df['X_token'].values.tolist()
    y = df['y'].values.tolist()
    # y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y,random_state = 1)

    count_vectorizer.fit(X_train)
    X_feature = count_vectorizer.transform(X_train)

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_feature, y_train)
    joblib.dump(clf, name_save)

    X_feature_test = count_vectorizer.transform(X_test)
    y_pred = clf.predict(X_feature_test)
    report = classification_report(y_test, y_pred)
    print(report)


# domain_full_file.xlsx
# df_domain = pd.read_excel(input_dir + r"\domain_full_file.xlsx")
# # sửa label __label__promotion, do chỉ có 1 instance
# # xem xét bỏ promotion đi, vì chỉ có 45 instance
# # df.loc[df['y']=='__label__promotion', 'y']= '__label__promotion '
# df_domain  = df_domain[df_domain['y']!='__label__promotion'].copy()
# train_save_model(df_domain,input_dir + r"\domain_full_file.pickle")
#
#
#
#
#
# # question_attribute_full_file
# df_attribute = pd.read_excel(input_dir + r"\question_attribute_full_file.xlsx")
# df_attribute = df_attribute[-df_attribute['X'].isnull()].copy()
# df_attribute  = df_attribute[-df_attribute['y'].isin(['__label__gender', '__label__type','__label__guarantee', '__label__origin'])].copy()
# train_save_model(df_attribute,input_dir + r"\question_attribute.pickle")
#

##### question_type_full_file
df_type = pd.read_excel(input_dir + r"\question_type_full_file.xlsx")
# xóa 1 instance NaN
df_type = df_type[-df_type['X'].isnull()].copy()
df_type  = df_type[df_type['y'] !="__label__chat"].copy()
train_save_model(df_type,input_dir + r"\question_type.pickle")
