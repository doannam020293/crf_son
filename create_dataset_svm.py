import os
from  pyvi import ViTokenizer
import pandas as pd


input_dir =  r'C:\Users\Windows 10 TIMT\OneDrive\Nam\OneDrive - Five9 Vietnam Corporation\work\data_input\nlp\chat bot\classification\data'

def apply_preprocess(sentence):
    new_sentence =  sentence.replace("\n",'')
    new_sentence = ViTokenizer.tokenize(new_sentence)
    return new_sentence

# read file
def write_full_file(file_dir,path_file_write):
    '''
    tổng hợp các file phân loại thành 1 file duy nhất.
    :return:
    '''
    X = []
    y = []
    file_writer = open(path_file_write + ".txt",'w',encoding='utf-8')
    for root, dirs, files in os.walk(file_dir, topdown=False):
       for name in files:
           file_path = os.path.join(root, name)
           with open(file_path,'r',encoding='utf-8') as file:
               for line in file:
                   try:
                        line_split = line.split(",",1)
                        assert(len(line_split)==2)
                        # if len(line_split) !=2:
                        #     print(line)
                        sentence  = line_split[1]
                        new_sentence = sentence.replace("\n", '')
                        new_sentence = new_sentence.strip()
                        # new_sentence = ViTokenizer.tokenize(new_sentence)
                        X.append(new_sentence)
                        y.append(line_split[0].strip())
                        file_writer.write(line)
                   except Exception as e:
                       print("eror {} tai line {}, ".format(e,line))
                       continue
    df = pd.DataFrame({"X":X,'y':y})
    df['X_token'] = df['X'].apply(apply_preprocess)
    df.to_excel(path_file_write+".xlsx",index=False)
    return

write_full_file(input_dir + r"\domain",input_dir + r"\domain_full_file")
write_full_file(input_dir + r"\question_attribute",input_dir + r"\question_attribute_full_file")
write_full_file(input_dir + r"\question_type",input_dir + r"\question_type_full_file")

