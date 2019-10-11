import re
'''
pattern1: 18 December 1869
pattern2: December 10 , 1860
pattern3: December 23
pattern4: December 1867
pattern5: 23 December
pattern6: 1860s
'''
pattern1 = r'\s\d{2}\s(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\s'
pattern2 = r'\s(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{2}\s,\s\d{4}\s'
pattern3 = r'\s(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{2}\s'
pattern4 = r'\s(January|February|March|April|May|June|July|August|September|October|November|December)\s\d{4}\s'
pattern5 = r'\s\d{2}\s(January|February|March|April|May|June|July|August|September|October|November|December)\s'
pattern6 = r'\s\d{4}s\s'

def replace_date(sentence):
    sentence = re.sub(pattern1, ' <day> <month> <year> ', sentence)
    sentence = re.sub(pattern2, ' <month> <day> , <year> ', sentence)
    sentence = re.sub(pattern3, ' <month> <day> ', sentence)
    sentence = re.sub(pattern4, ' <month> <year> ', sentence)
    sentence = re.sub(pattern5, ' <day> <month> ', sentence)
    sentence = re.sub(pattern6, ' <year> ', sentence)
    return sentence

def replace_number(sentence):
    sentence = re.sub(r'\s\d+\s',' <num> ',sentence)
    return sentence
    
def preprocess_file(file_path_in, file_path_out):
    fin = open(file_path_in, 'r')
    fout = open(file_path_out, 'w')     
    
    for sentence in fin:
        sentence = replace_date(sentence)
        sentence = replace_number(sentence)
        fout.write(sentence)
    fin.close()
    fout.close()      

if __name__ == "__main__":
    preprocess_file('data/wikitext-103/train.txt','data/wikitext-103/train_new.txt')
    
