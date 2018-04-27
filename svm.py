from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

import pandas as pd
import numpy as np

from sklearn import svm, cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split





def split_file(df,name_save, use_token= True):
    if use_token: # dùng n-gram đã qua tokeniser hay chưa
        X = df['X_token'].values.tolist()
    else:
        X = df['X'].values.tolist()
    y = df['y'].values.tolist()

    #stratify = y : phân chia tập train- test theo phân phối các label của y, tức là các nhãn của y sẽ có cùng 1 phân phối xác suất giống nhau trong tập train, và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 1)
    list_sentence = []
    with open(input_dir + r"\fastText\train_" + name_save  + ".txt",'w', encoding='utf-8', ) as file:
        for i, feature in enumerate(X_train):
            lablel = y_train[i]
            sentence = lablel + ' ' + feature + '\n'
            # list_sentence.append(sentence)
            file.write(sentence)
    with open(input_dir + r"\fastText\test_" +name_save  +  ".txt",'w', encoding='utf-8', ) as file:
        for i, feature in enumerate(X_test):
            lablel = y_test[i]
            sentence = lablel + ' ' + feature + '\n'
            # list_sentence.append(sentence)
            file.write(sentence)

def train_save_model(df,name_save, use_token= True,  use_tfidf= False, max_features = 10000):
    # dùng feature là bag of word với ngram là 1 và 2.
    # max_features đang để là 10.000 tương ứng với số lượng 1-gram (uni-gram) và 2-gram (2 cái này gọi chung là vocabulary) tối đa là 10.000. do chạy SVM rất tốn memory
    #min_df : min frequency 1 n-gram xuất hiện trong feature là 3.


    count_vectorizer = CountVectorizer(ngram_range=(1, 2),max_features= max_features, min_df=3) # đoạn này có thể dùng feature là tf-idf để nâng cao kết quả của model
    if use_tfidf:
        count_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=max_features,
                                           min_df=3)  # đoạn này có thể dùng feature là tf-idf để nâng cao kết quả của model
    if use_token: # dùng n-gram đã qua tokeniser hay chưa
        X = df['X_token'].values.tolist()
    else:
        X = df['X'].values.tolist()
    y = df['y'].values.tolist()

    #stratify = y : phân chia tập train- test theo phân phối các label của y, tức là các nhãn của y sẽ có cùng 1 phân phối xác suất giống nhau trong tập train, và test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 1)
    # train và fit model bag of n-gram
    count_vectorizer.fit(X_train)
    X_feature = count_vectorizer.transform(X_train)

    # decision_function_shape = ovr (one vs rest) do  bài toàn là multilabel, hơn nữa do mẫu khá lớn, (khá nhiều quan sát) nên để one-vs-rest cho đỡ tốn cost tính toán
    # ,class_weight="balanced". Do bài toán là imbalance, nên cần  đánh lại trọng số các label (các label có frequency thấp, sẽ được đánh trọng số cao hơn trong hàm mất mất (loss function))
    # đầu tiên thử kernel = linear . kernel = linear có chi phí tính toán thấp nhất trong các kernel, thì thấy kết quả khá tốt, nên không cần thử các kernel khác nữa.
    # Hơn nữa thời gian train 1 model mất 2-3 tiếng, nên cũng chỉ nên để kernel  = linear là hơp lý nhất.
    # Ngoài ra các tham số khác đều để mặc định.
    clf = svm.SVC(kernel='linear',class_weight="balanced",decision_function_shape='ovr')

    clf.fit(X_feature, y_train)
    joblib.dump(clf, name_save + "model.pickle")
    joblib.dump(count_vectorizer,  name_save + "transform_feature.pickle")

    X_feature_test = count_vectorizer.transform(X_test)
    y_pred = clf.predict(X_feature_test)
    report = classification_report(y_test, y_pred)
    print(report)


def predict_get_error(df, name_save):
    '''
    function này để dự đoán model đã được train, và tạo ra các file xlsx dự đoán sai
    :param df: 
    :param name_save: 
    :return: 
    '''
    X = df['X_token'].values.tolist()
    y = df['y'].values.tolist()
    # y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state = 1)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    clf = joblib.load( name_save + "model.pickle")
    count_vectorizer = joblib.load(  name_save + "transform_feature.pickle")
    X_feature_test = count_vectorizer.transform(X_test)
    y_pred = clf.predict(X_feature_test)
    misclassified_samples = X_test[y_test != y_pred]
    misclassified_predict = y_pred[y_test != y_pred]
    misclassified_true = y_test[y_test != y_pred]
    misclassified_df = pd.DataFrame({"token":misclassified_samples,"label": misclassified_true,"predict": misclassified_predict})
    misclassified_df.to_excel(name_save+"misclassified.xlsx",index=False)
    report = classification_report(y_test, y_pred)
    print(report)


def train_total():
    ##### question_type_full_file
    df_type = pd.read_excel(input_dir + r"\question_type_full_file.xlsx")
    """
    df_type['y'].value_counts()
    __label__attri     78739
    __label__exists    67483
    __label__yesno     38615
    __label__other     13366
    __label__diff      11374
    __label__when       4643
    __label__order      2317
    __label__chat         26
    
    """
    # xóa 1 instance NaN
    df_type = df_type[-df_type['X'].isnull()].copy()
    # df_type  = df_type[df_type['y'] !="__label__chat"].copy()
    train_save_model(df_type,input_dir + r"\question_ver3_type",use_token=False)
    predict_get_error(df_type, input_dir + r"\question_ver3_type")

    """
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
    """


    # domain_full_file.xlsx
    df_domain = pd.read_excel(input_dir + r"\domain_full_file.xlsx")
    """
    df_domain['y'].value_counts()
    __label__product      120719
    __label__ship          19770
    __label__refund         1868
    __label__shop            925
    __label__promotion        45
    """
    # df_domain  = df_domain[df_domain['y']!='__label__promotion'].copy()
    df_domain = df_domain[-df_domain['X'].isnull()].copy()
    train_save_model(df_domain,input_dir + r"\domain_ver3_full_file")
    predict_get_error(df_domain, input_dir + r"\domain_ver3_full_file")
    """
    
         
                        precision    recall  f1-score   support
    
      __label__product       1.00      1.00      1.00     24144
    __label__promotion       0.31      0.56      0.40         9
       __label__refund       0.99      0.98      0.99       374
         __label__ship       0.99      0.98      0.99      3954
         __label__shop       0.87      0.90      0.89       185
    
           avg / total       0.99      0.99      0.99     28666
    """


    # question_attribute_full_file
    df_attribute = pd.read_excel(input_dir + r"\question_attribute_full_file.xlsx")
    df_attribute = df_attribute[-df_attribute['X'].isnull()].copy()
    """
    df_attribute['y'].value_counts()
    __label__size         49415
    __label__price        45546
    __label__color         9407
    __label__weight        4796
    __label__promotion     3910
    __label__location      3214
    __label__time          2922
    __label__sale          1810
    __label__height        1312
    __label__material       909
    __label__trade          221
    __label__import         123
    __label__gender          65
    __label__type            15
    
    
    """
    # df_attribute  = df_attribute[-df_attribute['y'].isin(['__label__gender', '__label__type','__label__guarantee', '__label__origin'])].copy()
    train_save_model(df_attribute,input_dir + r"\question_ver3_attribute")
    predict_get_error(df_attribute, input_dir + r"\question_ver3_attribute")
    """
    
      precision    recall  f1-score   support
        __label__color       0.97      0.99      0.98      1881
       __label__gender       0.15      0.46      0.23        13
       __label__height       0.98      0.98      0.98       262
       __label__import       0.80      0.96      0.87        25
     __label__location       0.96      0.97      0.97       643
     __label__material       0.97      0.98      0.98       182
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




input_dir =  r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\data_input\nlp\chat bot\classification\data'

def run_split():
    df_type = pd.read_excel(input_dir + r"\question_type_full_file.xlsx")
    df_type = df_type[-df_type['X'].isnull()].copy()

    split_file(df_type,'df_type', use_token= True)



    # domain_full_file.xlsx
    df_domain = pd.read_excel(input_dir + r"\domain_full_file.xlsx")
    df_domain = df_domain[-df_domain['X'].isnull()].copy()
    split_file(df_domain,'df_domain', use_token= True)

    # question_attribute_full_file
    df_attribute = pd.read_excel(input_dir + r"\question_attribute_full_file.xlsx")
    df_attribute = df_attribute[-df_attribute['X'].isnull()].copy()
    split_file(df_attribute,'df_attribute', use_token= True)


def train_total_ver2():
    ##### question_type_full_file
    df_type = pd.read_excel(input_dir + r"\question_type_full_file.xlsx")
    """
    df_type['y'].value_counts()
    __label__attri     78739
    __label__exists    67483
    __label__yesno     38615
    __label__other     13366
    __label__diff      11374
    __label__when       4643
    __label__order      2317
    __label__chat         26

    """
    # xóa 1 instance NaN
    df_type = df_type[-df_type['X'].isnull()].copy()
    # df_type  = df_type[df_type['y'] !="__label__chat"].copy()
    train_save_model(df_type, input_dir + r"\question_ver3_type", use_token=False)
    # predict_get_error(df_type, input_dir + r"\question_ver3_type")

    """
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
    """

    # domain_full_file.xlsx
    df_domain = pd.read_excel(input_dir + r"\domain_full_file.xlsx")
    """
    df_domain['y'].value_counts()
    __label__product      120719
    __label__ship          19770
    __label__refund         1868
    __label__shop            925
    __label__promotion        45
    """
    # df_domain  = df_domain[df_domain['y']!='__label__promotion'].copy()
    df_domain = df_domain[-df_domain['X'].isnull()].copy()
    train_save_model(df_domain, input_dir + r"\domain_ver3_full_file",use_token=False)
    # predict_get_error(df_domain, input_dir + r"\domain_ver3_full_file")
    """


                        precision    recall  f1-score   support

      __label__product       1.00      1.00      1.00     24144
    __label__promotion       0.31      0.56      0.40         9
       __label__refund       0.99      0.98      0.99       374
         __label__ship       0.99      0.98      0.99      3954
         __label__shop       0.87      0.90      0.89       185

           avg / total       0.99      0.99      0.99     28666
    """

    # question_attribute_full_file
    df_attribute = pd.read_excel(input_dir + r"\question_attribute_full_file.xlsx")
    df_attribute = df_attribute[-df_attribute['X'].isnull()].copy()
    """
    df_attribute['y'].value_counts()
    __label__size         49415
    __label__price        45546
    __label__color         9407
    __label__weight        4796
    __label__promotion     3910
    __label__location      3214
    __label__time          2922
    __label__sale          1810
    __label__height        1312
    __label__material       909
    __label__trade          221
    __label__import         123
    __label__gender          65
    __label__type            15


    """
    # df_attribute  = df_attribute[-df_attribute['y'].isin(['__label__gender', '__label__type','__label__guarantee', '__label__origin'])].copy()
    train_save_model(df_attribute, input_dir + r"\question_ver3_attribute",use_token=False)
    # predict_get_error(df_attribute, input_dir + r"\question_ver3_attribute")
    """

      precision    recall  f1-score   support
        __label__color       0.97      0.99      0.98      1881
       __label__gender       0.15      0.46      0.23        13
       __label__height       0.98      0.98      0.98       262
       __label__import       0.80      0.96      0.87        25
     __label__location       0.96      0.97      0.97       643
     __label__material       0.97      0.98      0.98       182
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
def result_run_ver2():
    train_total_ver2()
    """
                     precision    recall  f1-score   support

     __label__attri       0.99      0.98      0.99     15748
      __label__chat       0.75      0.60      0.67         5
      __label__diff       0.94      0.96      0.95      2275
    __label__exists       0.97      0.95      0.96     13497
     __label__order       0.99      0.99      0.99       463
     __label__other       0.96      0.97      0.96      2673
      __label__when       0.98      0.98      0.98       929
     __label__yesno       0.93      0.95      0.94      7723

        avg / total       0.97      0.97      0.97     43313

                        precision    recall  f1-score   support

      __label__product       1.00      1.00      1.00     24144
    __label__promotion       0.25      0.33      0.29         9
       __label__refund       0.99      1.00      1.00       374
         __label__ship       1.00      1.00      1.00      3954
         __label__shop       0.88      0.93      0.91       185

           avg / total       1.00      1.00      1.00     28666

                        precision    recall  f1-score   support
    C:\ProgramData\Miniconda3\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.

      'precision', 'predicted', average, warn_for)
        __label__color       0.99      1.00      0.99      1881
       __label__gender       0.82      0.69      0.75        13
    __label__guarantee       0.00      0.00      0.00         1
       __label__height       0.98      1.00      0.99       262
       __label__import       0.88      0.92      0.90        25
     __label__location       0.98      0.98      0.98       643
     __label__material       1.00      0.99      1.00       182
       __label__origin       0.00      0.00      0.00         1
        __label__price       1.00      1.00      1.00      9109
    __label__promotion       0.99      0.99      0.99       782
         __label__sale       0.98      0.99      0.99       362
         __label__size       1.00      1.00      1.00      9883
         __label__time       0.99      0.99      0.99       584
        __label__trade       0.93      0.95      0.94        44
         __label__type       1.00      1.00      1.00         3
       __label__weight       0.99      0.99      0.99       959

           avg / total       1.00      1.00      1.00     24734
    """
def train_ver3():

    ##### question_type_full_file
    df_type = pd.read_excel(input_dir + r"\question_type_full_file.xlsx")
    """
    df_type['y'].value_counts()
    __label__attri     78739
    __label__exists    67483
    __label__yesno     38615
    __label__other     13366
    __label__diff      11374
    __label__when       4643
    __label__order      2317
    __label__chat         26

    """
    # xóa 1 instance NaN
    df_type = df_type[-df_type['X'].isnull()].copy()
    # df_type  = df_type[df_type['y'] !="__label__chat"].copy()
    train_save_model(df_type, input_dir + r"\question_ver3_type", use_token=False,use_tfidf=True)
    # predict_get_error(df_type, input_dir + r"\question_ver3_type")

    """
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
    """

    # domain_full_file.xlsx
    df_domain = pd.read_excel(input_dir + r"\domain_full_file.xlsx")
    """
    df_domain['y'].value_counts()
    __label__product      120719
    __label__ship          19770
    __label__refund         1868
    __label__shop            925
    __label__promotion        45
    """
    # df_domain  = df_domain[df_domain['y']!='__label__promotion'].copy()
    df_domain = df_domain[-df_domain['X'].isnull()].copy()
    train_save_model(df_domain, input_dir + r"\domain_ver3_full_file", use_token=False,use_tfidf=True)
    # predict_get_error(df_domain, input_dir + r"\domain_ver3_full_file")
    """


                        precision    recall  f1-score   support

      __label__product       1.00      1.00      1.00     24144
    __label__promotion       0.31      0.56      0.40         9
       __label__refund       0.99      0.98      0.99       374
         __label__ship       0.99      0.98      0.99      3954
         __label__shop       0.87      0.90      0.89       185

           avg / total       0.99      0.99      0.99     28666
    """

    # question_attribute_full_file
    df_attribute = pd.read_excel(input_dir + r"\question_attribute_full_file.xlsx")
    df_attribute = df_attribute[-df_attribute['X'].isnull()].copy()
    """
    df_attribute['y'].value_counts()
    __label__size         49415
    __label__price        45546
    __label__color         9407
    __label__weight        4796
    __label__promotion     3910
    __label__location      3214
    __label__time          2922
    __label__sale          1810
    __label__height        1312
    __label__material       909
    __label__trade          221
    __label__import         123
    __label__gender          65
    __label__type            15


    """
    # df_attribute  = df_attribute[-df_attribute['y'].isin(['__label__gender', '__label__type','__label__guarantee', '__label__origin'])].copy()
    train_save_model(df_attribute, input_dir + r"\question_ver3_attribute", use_token=False, use_tfidf=True)
    # predict_get_error(df_attribute, input_dir + r"\question_ver3_attribute")
    """

      precision    recall  f1-score   support
        __label__color       0.97      0.99      0.98      1881
       __label__gender       0.15      0.46      0.23        13
       __label__height       0.98      0.98      0.98       262
       __label__import       0.80      0.96      0.87        25
     __label__location       0.96      0.97      0.97       643
     __label__material       0.97      0.98      0.98       182
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
def run_train_ver3():
    """
      "This module will be removed in 0.20.", DeprecationWarning)
                 precision    recall  f1-score   support

 __label__attri       0.99      0.98      0.99     15748
  __label__chat       1.00      0.60      0.75         5
  __label__diff       0.89      0.98      0.93      2275
__label__exists       0.97      0.94      0.96     13497
 __label__order       0.99      1.00      1.00       463
 __label__other       0.96      0.97      0.96      2673
  __label__when       0.94      1.00      0.97       929
 __label__yesno       0.92      0.95      0.94      7723

    avg / total       0.96      0.96      0.96     43313

                    precision    recall  f1-score   support

  __label__product       1.00      1.00      1.00     24144
__label__promotion       0.14      0.44      0.21         9
   __label__refund       0.99      1.00      1.00       374
     __label__ship       1.00      1.00      1.00      3954
     __label__shop       0.88      0.95      0.92       185

       avg / total       1.00      1.00      1.00     28666

                    precision    recall  f1-score   support
C:\ProgramData\Miniconda3\lib\site-packages\sklearn\metrics\classification.py:1135: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.

  'precision', 'predicted', average, warn_for)
    __label__color       0.97      1.00      0.98      1881
   __label__gender       0.78      0.54      0.64        13
__label__guarantee       0.00      0.00      0.00         1
   __label__height       0.98      0.98      0.98       262
   __label__import       0.88      0.88      0.88        25
 __label__location       0.96      0.99      0.97       643
 __label__material       1.00      0.99      1.00       182
   __label__origin       0.00      0.00      0.00         1
    __label__price       0.99      1.00      1.00      9109
__label__promotion       0.99      0.98      0.99       782
     __label__sale       0.99      0.98      0.99       362
     __label__size       1.00      0.99      0.99      9883
     __label__time       0.99      0.99      0.99       584
    __label__trade       0.98      0.93      0.95        44
     __label__type       1.00      1.00      1.00         3
   __label__weight       0.99      0.99      0.99       959

       avg / total       0.99      0.99      0.99     24734


Process finished with exit code 0
    :return:
    """
    pass