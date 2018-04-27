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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    count_vectorizer.fit(X_train)
    X_feature = count_vectorizer.transform(X_train)

    clf = svm.SVC(kernel='linear', probability=True)
    clf.fit(X_feature, y_train)
    joblib.dump(clf, name_save)

    X_feature_test = count_vectorizer.transform(X_test)
    y_pred = clf.predict(X_feature_test)
    report = classification_report(y_test, y_pred)
    print(report)


# # domain_full_file.xlsx
# df_domain = pd.read_excel(input_dir + r"\domain_full_file.xlsx")
# # sửa label __label__promotion, do chỉ có 1 instance
# # xem xét bỏ promotion đi, vì chỉ có 45 instance
# # df.loc[df['y']=='__label__promotion', 'y']= '__label__promotion '
# df_domain  = df_domain[df_domain['y']!='__label__promotion'].copy()
# train_save_model(df_domain,input_dir + r"\domain_full_file.pickle")
#
#



# question_attribute_full_file
df_attribute = pd.read_excel(input_dir + r"\question_attribute_full_file.xlsx")
df_attribute = df_attribute[-df_attribute['X'].isnull()].copy()
df_attribute  = df_attribute[-df_attribute['y'].isin(['__label__gender', '__label__type','__label__guarantee', '__label__origin'])].copy()
train_save_model(df_attribute,input_dir + r"\question_attribute.pickle")

#
# ##### question_type_full_file
# df_type = pd.read_excel(input_dir + r"\question_type_full_file.xlsx")
# # xóa 1 instance NaN
# df_type = df_type[-df_type['X'].isnull()].copy()
# df_type  = df_type[df_attribute['y'] !="__label__chat"].copy()
# train_save_model(df_type,input_dir + r"\question_type.pickle")
"""
#                   precision    recall  f1-score   support
#
# __label__product       1.00      1.00      1.00     24144
#  __label__refund       0.98      0.99      0.99       374
#    __label__ship       1.00      0.99      0.99      3954
#    __label__shop       0.91      0.92      0.92       185
#
#      avg / total       1.00      1.00      1.00     28657
#

               precision    recall  f1-score   support

__label__product       1.00      1.00      1.00     24144
 __label__refund       0.99      0.99      0.99       374
   __label__ship       1.00      0.99      0.99      3954
   __label__shop       0.95      0.94      0.94       185

     avg / total       1.00      1.00      1.00     28657
     
     
                       precision    recall  f1-score   support

 __label__attri       0.99      0.98      0.99     15748
  __label__chat       0.75      0.60      0.67         5
  __label__diff       0.93      0.94      0.94      2275
__label__exists       0.96      0.96      0.96     13497
 __label__order       0.98      0.98      0.98       463
 __label__other       0.94      0.96      0.95      2673
  __label__when       0.97      0.97      0.97       929
 __label__yesno       0.93      0.95      0.94      7723

    avg / total       0.96      0.96      0.96     43313

                    precision    recall  f1-score   support

  __label__product       1.00      1.00      1.00     24144
__label__promotion       0.31      0.56      0.40         9
   __label__refund       0.99      0.98      0.99       374
     __label__ship       0.99      0.98      0.99      3954
     __label__shop       0.87      0.90      0.89       185

       avg / total       0.99      0.99      0.99     28666

                    precision    recall  f1-score   support
C:\ProgramData\Miniconda3\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.

    __label__color       0.97      0.99      0.98      1881
  'precision', 'predicted', average, warn_for)
   __label__gender       0.15      0.46      0.23        13
__label__guarantee       0.00      0.00      0.00         1
   __label__height       0.98      0.98      0.98       262
   __label__import       0.80      0.96      0.87        25
 __label__location       0.96      0.97      0.97       643
 __label__material       0.97      0.98      0.98       182
   __label__origin       0.00      0.00      0.00         1
    __label__price       0.99      0.99      0.99      9109
__label__promotion       0.99      0.97      0.98       782
     __label__sale       0.98      0.97      0.98       362
     __label__size       1.00      0.99      0.99      9883
     __label__time       0.99      0.98      0.98       584
    __label__trade       0.91      0.89      0.90        44
     __label__type       1.00      1.00      1.00         3
   __label__weight       0.99      0.99      0.99       959

       avg / total       0.99      0.99      0.99     24734
"""