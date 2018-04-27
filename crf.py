import string
import os
import sys
import pickle
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import numpy as np
import pandas as pd
# read file
def read_file():
    '''
    đọc các file txt, ghi vào biến corpưs, sau đó lưu ra file .pickle
    mỗi câu là 1 list,trong list này thì vị trí đầu tiên : token, vị trị trí thứ 2: POS, thứ 3: Chunk, thứ 4: nhãn NER, thứ 5: nhãn label
    :return:
    '''
    corpus = []
    ############ thay đổi đường dẫn đến folder chứa file txt
    file_dir = '/home/nam/Desktop/data_seq'
    for root, dirs, files in os.walk(file_dir, topdown=False):
        for name in files:
            file_path = os.path.join(root, name)
            # file_path = '/home/nam/Desktop/data_seq/417403_438082.txt'
            with open(file_path, 'r', encoding='utf-8') as file:
                sentence = []
                for line in file:
                    # line = 'bạn	N	B-NP	O	O'
                    line = line.replace('\n', "")
                    if line != '':

                        line = line.split('\t')
                        # thay thế khoảng trắng giữa các từ bằng dấu gạch dưới
                        line[0] = line[0].replace(" ", "_")
                        sentence.append(line)
                    else:
                        corpus.append(sentence)
                        sentence = []
    ############ thay đổi đường dẫn lưu file
    with open('/home/nam/Desktop/data_seq/copus.pickle', 'wb') as file:
        pickle.dump(corpus, file)
    return corpus


def word2features(sent, i, is_training):
    '''
    tạo feature cho từng word
    1 word sẽ lấy feature là chính nó, và 2 từ đằng trước, 2 từ đằng sau
    với 1 từ sẽ lấy các feature:
     chính  từ đó.
     có viết thường toàn bộ  hay k
     có viết hoa âm đầu hay k
     có viết hoa toàn bộ hay k
     có là số hay không
     có nằm trong các filter dấu biểu cảm hay k
    :param sent: dạng type là list, chứa token và nhãn của 1 câu
    :param i: từ đó vị trí thứ i trong câu
    :param is_training: đây là train hay test, nếu là test thì ta chỉ có list các token, k có l
    :return:abel
    '''
    # filtered_tags l các kí tự đặc biết
    filtered_tags = set(string.punctuation)
    filtered_tags.add(u'\u2026')
    filtered_tags.add(u'\u201d')
    filtered_tags.add(u'\u201c')
    filtered_tags.add(u'\u2019')
    filtered_tags.add('...')
    # lấy từ đó
    word = sent[i][0] if is_training else sent[i]

    # feature tương ứng với chính từ đó
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        # 'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word[:1].isdigit()': word[:1].isdigit(),
        'word[:3].isupper()': word[:3].isupper(),
        #       'word.indict()': word in vi_words,
        'word.isfiltered': word in filtered_tags,
    }
    # nếu i >0, thì ta có thêm feature của từ liền trước
    if i > 0:
        word1 = sent[i - 1][0] if is_training else sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word[:1].isdigit()': word1[:1].isdigit(),
            '-1:word[:3].isupper()': word1[:3].isupper(),
        })
        # nếu i >1, thì ta có thêm feature từ trước đó 2 t
        if i > 1:
            word2 = sent[i - 2][0] if is_training else sent[i - 2]
            features.update({
                '-2:word.lower()': word2.lower(),
                '-2:word.istitle()': word2.istitle(),
                '-2:word.isupper()': word2.isupper(),
            })
    else:
        # i = 0, thì add thêm feature là begin of sentence
        features['BOS'] = True
    # nếu i < len(sent) - 1, thì ta có thêm feature của từ liền sau
    if i < len(sent) - 1:
        word1 = sent[i + 1][0] if is_training else sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word[:1].isdigit()': word1[:1].isdigit(),
            '+1:word.isupper()': word1.isupper(),
        })
        # nếu i< len(sent) - 2, thì ta có thêm feature của từ  sau đó 2 từ
        if i < len(sent) - 2:
            word2 = sent[i + 2][0] if is_training else sent[i + 2]
            features.update({
                '+2:word.lower()': word2.lower(),
                '+2:word.istitle()': word2.istitle(),
                '+2:word.isupper()': word2.isupper(),
            })
    else:
        # nếu k add thêm feature end of sentence
        features['EOS'] = True

    return features


def sent2features(sent, is_training=True):
    '''
    lấy feature cho toàn bộ 1 câu
    :param sent: là 1 list các từ trong câu đã được tách từ
    :param is_training:
    :return:
    '''

    return [word2features(sent, i, is_training) for i in range(len(sent))]


def sent2labels(sent, i):
    '''
    lấy label cho 1 câu
    :param sent:
    :param i: vị trí label muốn lấy
    :return:
    '''
    return [a[i] for a in sent]


# sentence = [a[0] for a in corpus[0]]

def get_counter(y_ner):
    '''
    kiểm tra số lần xuất hiện các nhãn
    :param y_ner:
    :return:
    '''
    a = [a for y in y_ner for a in y]
    x = Counter(a)
    print(x)


def split_train_validation(X, y, test_size=0.2):
    '''
    chia tập train, test theo tỷ lệ  test_size tương  ứng
    :param X:
    :param y:
    :param test_size:
    :return:
    '''
    # chia tập train, test theo tỷ lệ 80-20. Do 1 số nhãn k chỉ có 1 instance,
    # nên  ta k chia tập train-test sao cho phân bổ các nhãn y là như nhau được stratify= y bị error
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42) # stratify = y
    return X_train, X_test, y_train, y_test


def train(X_train, X_test, y_train, y_test, name_save):
    """
    train model
    :param X:
    :param y:
    :param name_save: tên model save file pickle
    :return:
    """
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2 , random_state=42)

    # khởi tạo model CRF,
    # thuật toán Gradient descent using the L-BFGS method
    # trọng số regularization L1 trong hàm loss là 0.1
    # trọng số regularization L2 trong hàm loss là 0.1
    # tính toàn bộ các transition giữa cac label
    crf = sklearn_crfsuite.CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    # fit model
    crf.fit(X_train, y_train)

    #  tính toán và print accuracy

    # lấy các nhãn trong tập train
    labels = list(crf.classes_)
    labels.remove('O')
    # labels.remove('o')
    print(labels)
    # predict nhãn cho tập test
    y_pred = crf.predict(X_test)
    # tính accuracy
    metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))
    # ghi file pickle
    joblib.dump(crf, name_save)





def get_accuracy(X_test, y_test, file_model):
    '''
    print accuracy của tập test tương ứng với file_model.
    :param X_test:
    :param y_test:
    :param file_model:
    :return:
    '''
    crf = joblib.load(file_model)
    labels = list(crf.classes_)
    list_remove = ['O`','O','o', 'I-MATE', 'B-CURU', 'I-ORG', 'I-GEND']
    for a in list_remove:
        if a  in labels: labels.remove(a)

    # predict nhãn cho tập test
    y_pred = crf.predict(X_test)

    X_test = np.array(X_test)
    y_test = np.array(y_test)
    y_pred = np.array(y_pred)

    misclassified_samples_raw = X_test[y_test != y_pred]
    misclassified_samples = []
    for a in list(misclassified_samples_raw):
        # a = list(misclassified_samples)[0]
        x =[ b['word.lower()'] for b in a ]
        sentence = ' '.join(x)
        misclassified_samples.append(sentence)

    misclassified_predict = y_pred[y_test != y_pred]
    misclassified_true = y_test[y_test != y_pred]
    misclassified_df = pd.DataFrame({"token":misclassified_samples,"label": misclassified_true,"predict": misclassified_predict})
    misclassified_df.to_excel(file_model+"misclassified.xlsx",index=False)
    # tính accuracy
    metrics.flat_f1_score(y_test, y_pred,
                          average='weighted', labels=labels)

    sorted_labels = sorted(
        labels,
        key=lambda name: (name[1:], name[0])
    )
    print(metrics.flat_classification_report(
        y_test, y_pred, labels=sorted_labels, digits=3
    ))


def train_full():
    input_file = r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\data_input\nlp\chat bot\entity'
    with open(input_file + r'\copus.pickle', 'rb') as file:
        corpus = pickle.load(file)
    # lấy giá trị của X input
    X = [sent2features(s) for s in corpus]
    # lấy nhãn pos
    y_pos = [sent2labels(s, 1) for s in corpus]
    """
    Counter({'N': 623218, 'V': 389616, 'R': 261614, 'P': 250741, 'T': 137928, 'CH': 135466, 'A': 92576, 'M': 86171, 'Nc': 77179, 'E': 66693, 'I': 64437, 'C': 58421, 'Np': 33351, 'L': 8905, 'X': 3374, 'Nu': 2513, 'FW': 1283, 'Ny': 714, 'Z': 345, 'O': 12})
    """
    # split file train, test
    X_train, X_test, y_pos_train, y_pos_test = split_train_validation(X, y_pos)
    # train model và lưu file pickle
    ############ muốn train lại thì thay đổi đường dẫn lưu file, và bỏ comment câu sau
    # train(X_train, X_test, y_pos_train, y_pos_test,'/home/nam/Desktop/data_seq/model_pos.pickle')

    # lấy nhãn chunk
    y_chunk = [sent2labels(s, 2) for s in corpus]
    """
    Counter({'B-NP': 1070551, 'O': 662302, 'B-VP': 397868, 'B-AP': 94275, 'B-PP': 66693, 'I-NP': 2868})
    """
    # split file train, test
    X_train, X_test, y_chunk_train, y_chunk_test = split_train_validation(X, y_chunk)
    # train model và lưu file pickle
    ############ muốn train lại thì thay đổi đường dẫn lưu file, và bỏ comment câu sau
    # train(X_train, X_test, y_chunk_train, y_chunk_test,'/home/nam/Desktop/data_seq/model_chunk.pickle')

    # lấy nhãn ner
    y_ner = [sent2labels(s, 3) for s in corpus]
    """
    Counter({'O': 1993498, 'B-REFE': 63194, 'I-REFE': 45752, 'B-TYPE': 45735, 'B-SIZE': 29554, 'B-COLO': 22797, 'I-LOC': 14751, 'B-LOC': 14108, 'B-TIME': 11955, 'B-WEIG': 11059, 'B-MATE': 9555, 'B-PRIC': 8895, 'B-HEIG': 7324, 'B-TRAD': 4616, 'B-SAOF': 2662, 'I-SAOF': 2662, 'I-WEIG': 1876, 'I-COLO': 1047, 'I-SIZE': 697, 'B-GEND': 489, 'I-SHME': 454, 'B-ORIG': 378, 'B-SHME': 355, 'I-TYPE': 336, 'I-HEIG': 324, 'I-TIME': 189, 'I-MISC': 116, 'B-MISC': 108, 'I-TRAD': 40, 'I-ORIG': 16, 'B-ORG': 10, 'I-MATE': 1, 'B-CURU': 1, 'I-ORG': 1, 'o': 1, 'I-GEND': 1})
    """
    # split file train, test
    X_train, X_test, y_ner_train, y_ner_test = split_train_validation(X, y_ner)
    # train model và lưu file pickle
    ############ muốn train lại thì thay đổi đường dẫn lưu file, và bỏ comment câu sau
    # train(X_train, X_test, y_ner_train, y_ner_test,'/home/nam/Desktop/data_seq/y_ner.pickle')

    # lấy nhãn label2
    y_label2 = [sent2labels(s, 4) for s in corpus]
    # split file train, test
    X_train, X_test, y_label2_train, y_label2_test = split_train_validation(X, y_label2)
    """
    Counter({'O': 2260102, 'B-TYPE': 34234, 'B-MATE': 123, 'B-COLO': 84, 'B-GEND': 13, 'O`': 1})
    """
    # train model và lưu file pickle
    ############ muốn train lại thì thay đổi đường dẫn lưu file, và bỏ comment câu sau
    # train(X_train, X_test, y_label2_train, y_label2_test,'/home/nam/Desktop/data_seq/y_label2.pickle')


    # print kết quả model đã train
    get_accuracy(X_test, y_pos_test, input_file + r'\model_pos.pickle')

    """
        Pos
                    precision    recall  f1-score   support
              A      0.987     0.979     0.983     18360
              C      0.984     0.981     0.983     11746
              E      0.983     0.986     0.984     13228
              I      0.976     0.983     0.980     12836
              L      0.994     0.989     0.992      1846
              M      0.988     0.989     0.989     17203
              N      0.986     0.990     0.988    124278
              P      0.992     0.994     0.993     50138
              R      0.989     0.988     0.989     52005
              T      0.981     0.974     0.978     27782
              V      0.978     0.975     0.976     78228
              X      0.976     0.944     0.960       697
              Z      0.987     1.000     0.994        77
             CH      1.000     1.000     1.000     26802
             FW      0.981     0.958     0.969       264
             Nc      0.993     0.995     0.994     15373
             Np      0.973     0.967     0.970      6614
             Nu      0.964     0.968     0.966       528
             Ny      0.972     0.920     0.945       112
    avg / total      0.986     0.986     0.986    458117

    """
    get_accuracy(X_test, y_chunk_test, input_file + r'\model_chunk.pickle')

    """
    chunk
        ['B-NP', 'B-VP', 'B-PP', 'B-AP', 'I-NP']
                 precision    recall  f1-score   support
           B-AP      0.971     0.954     0.962     18705
           B-NP      0.974     0.983     0.978    213628
           I-NP      0.874     0.793     0.832       570
           B-PP      0.968     0.975     0.971     13228
           B-VP      0.949     0.939     0.944     79894
    avg / total      0.967     0.970     0.968    326025

    """
    get_accuracy(X_test, y_ner_test, input_file+ r'\y_ner.pickle')
    """
    B-CURU     I-GEND I-MATE B-ORG
        ner
                     precision    recall  f1-score   support
              o      0.000     0.000     0.000         1
         B-COLO      0.999     0.994     0.997      4556
         I-COLO      0.995     0.986     0.990       210
         B-CURU      0.000     0.000     0.000         0
         B-GEND      1.000     0.991     0.995       109
         I-GEND      0.000     0.000     0.000         0
         B-HEIG      1.000     0.992     0.996      1456
         I-HEIG      1.000     1.000     1.000        52
          B-LOC      0.918     0.810     0.860      2734
          I-LOC      0.921     0.813     0.864      2856
         B-MATE      1.000     0.996     0.998      1985
         I-MATE      0.000     0.000     0.000         0
         B-MISC      1.000     0.882     0.938        17
         I-MISC      1.000     0.950     0.974        20
          B-ORG      1.000     1.000     1.000         1
          I-ORG      0.000     0.000     0.000         0
         B-ORIG      1.000     0.969     0.984        64
         I-ORIG      1.000     1.000     1.000         3
         B-PRIC      0.999     0.998     0.999      1801
         B-REFE      0.995     0.999     0.997     12609
         I-REFE      1.000     0.999     1.000      9033
         B-SAOF      1.000     1.000     1.000       553
         I-SAOF      1.000     1.000     1.000       553
         B-SHME      1.000     1.000     1.000        76
         I-SHME      1.000     1.000     1.000       108
         B-SIZE      0.995     1.000     0.997      5901
         I-SIZE      0.993     0.993     0.993       138
         B-TIME      1.000     0.996     0.998      2318
         I-TIME      1.000     1.000     1.000        40
         B-TRAD      1.000     0.994     0.997       936
         I-TRAD      1.000     1.000     1.000         8
         B-TYPE      1.000     0.993     0.996      9191
         I-TYPE      0.982     0.982     0.982        57
         B-WEIG      1.000     1.000     1.000      2198
         I-WEIG      1.000     1.000     1.000       373
        avg / total      0.991     0.980     0.985     59957

    """
    get_accuracy(X_test, y_label2_test,input_file +  r'\y_label2.pickle')

    """
    mate: loại vải
    COLO: màu
    GEND: giới tính
    MATE: loại vải
    TYPE: quần/ áo 
        label2
                     precision    recall  f1-score   support
         B-COLO      1.000     0.667     0.800         9
         B-GEND      1.000     1.000     1.000         2
         B-MATE      1.000     0.952     0.976        21
         B-TYPE      0.989     1.000     0.994      6756
             O`      0.000     0.000     0.000         0
        avg / total      0.989     0.999     0.994      6788

    """
    """
    total
                 precision    recall  f1-score   support

          A      0.987     0.979     0.983     18360
          C      0.984     0.981     0.983     11746
          E      0.983     0.986     0.984     13228
          I      0.976     0.983     0.980     12836
          L      0.994     0.989     0.992      1846
          M      0.988     0.989     0.989     17203
          N      0.986     0.990     0.988    124278
          P      0.992     0.994     0.993     50138
          R      0.989     0.988     0.989     52005
          T      0.981     0.974     0.978     27782
          V      0.978     0.975     0.976     78228
          X      0.976     0.944     0.960       697
          Z      0.987     1.000     0.994        77
         CH      1.000     1.000     1.000     26802
         FW      0.981     0.958     0.969       264
         Nc      0.993     0.995     0.994     15373
         Np      0.973     0.967     0.970      6614
         Nu      0.964     0.968     0.966       528
         Ny      0.972     0.920     0.945       112

avg / total      0.986     0.986     0.986    458117

             precision    recall  f1-score   support

       B-AP      0.971     0.954     0.962     18705
       B-NP      0.974     0.983     0.978    213628
       I-NP      0.874     0.793     0.832       570
       B-PP      0.968     0.975     0.971     13228
       B-VP      0.949     0.939     0.944     79894

avg / total      0.967     0.970     0.968    326025

             precision    recall  f1-score   support

     B-COLO      0.999     0.994     0.997      4556
     I-COLO      0.995     0.986     0.990       210
     B-GEND      1.000     0.991     0.995       109
     B-HEIG      1.000     0.992     0.996      1456
     I-HEIG      1.000     1.000     1.000        52
      B-LOC      0.918     0.810     0.860      2734
      I-LOC      0.921     0.813     0.864      2856
     B-MATE      1.000     0.996     0.998      1985
     B-MISC      1.000     0.882     0.938        17
     I-MISC      1.000     0.950     0.974        20
      B-ORG      1.000     1.000     1.000         1
     B-ORIG      1.000     0.969     0.984        64
     I-ORIG      1.000     1.000     1.000         3
     B-PRIC      0.999     0.998     0.999      1801
     B-REFE      0.995     0.999     0.997     12609
     I-REFE      1.000     0.999     1.000      9033
     B-SAOF      1.000     1.000     1.000       553
     I-SAOF      1.000     1.000     1.000       553
     B-SHME      1.000     1.000     1.000        76
     I-SHME      1.000     1.000     1.000       108
     B-SIZE      0.995     1.000     0.997      5901
     I-SIZE      0.993     0.993     0.993       138
     B-TIME      1.000     0.996     0.998      2318
     I-TIME      1.000     1.000     1.000        40
     B-TRAD      1.000     0.994     0.997       936
     I-TRAD      1.000     1.000     1.000         8
     B-TYPE      1.000     0.993     0.996      9191
     I-TYPE      0.982     0.982     0.982        57
     B-WEIG      1.000     1.000     1.000      2198
     I-WEIG      1.000     1.000     1.000       373

avg / total      0.991     0.980     0.985     59956

             precision    recall  f1-score   support

     B-COLO      1.000     0.667     0.800         9
     B-GEND      1.000     1.000     1.000         2
     B-MATE      1.000     0.952     0.976        21
     B-TYPE      0.989     1.000     0.994      6756

avg / total      0.989     0.999     0.994      6788

    """

def predict(sentence, file_model):
    '''
    dự đoán nhãn của 1 câu
    :param X: feature của 1 câu, là output của sent2features()
    :param file_model:
    :return:
    '''
    sentence_split = sentence.split(" ")
    feature_x = [sent2features(sentence_split, is_training=False),]
    crf = joblib.load(file_model)
    y = crf.predict(feature_x)[0]
    return y


#
# sentence = "áo còn không cửa_hàng"
#
# input_path = r"C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\data_input\nlp\chat bot"
# model_name_crf =input_path +  r"\entity\y_label2.pickle"
# crf = joblib.load( model_name_crf)
# crf.classes_
#
# model_name_clf = input_path+ r"\classification\domain_ver3_full_filemodel.pickle"
# """
# __label__product      120719
# __label__ship          19770
# __label__refund         1868
# __label__shop            925
# __label__promotion        45
#
# """
# model_name_transform =input_path+  r"\classification\domain_ver3_full_filetransform_feature.pickle"
# clf = joblib.load(model_name_clf)
# transform = joblib.load( model_name_transform)
# count_vectorizer = joblib.load(name_save + "transform_feature.pickle")
#
#
#
#
# # with open(input_path + r'/copus.pickle', 'rb') as file:
# #     corpus = pickle.load(file)
# #
# # s = corpus[0]
# # X = sent2features(s)
# #
# # X1 = [sent2features(s) for s in corpus]
# # feature_x = X1[0]
# # feature_x = [feature_x,]
# # X = sent2features(sentence, False)
# #
# # crf = joblib.load(input_path+ model_name)
# # y = crf.predict(feature_x)
# #
#
# predict(sentence,input_path+ model_name)
train_full()