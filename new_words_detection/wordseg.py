# -*- coding:utf-8 -*-
"""
Chinese word segmentation algorithm with corpus
Author: "Xylander"
"""

import os
import re
import math
import time
from entropy import compute_entropy
from extract import extract_cadicateword, gen_bigram
import pandas as pd
import jieba
import pkuseg
from tqdm import tqdm


class wordinfo(object):
    '''
    Record every candidate word information include left neighbors, right neighbors, frequency, PMI
    '''

    def __init__(self, text):
        super(wordinfo, self).__init__()
        self.text = text
        self.freq = 0.0
        self.left = []  # record left neighbors
        self.right = []  # record right neighbors
        self.pmi = 0
        self.length = 0

    def update_data(self, left, right):
        self.freq += 1.0
        if left:
            self.left.append(left)
        if right:
            self.right.append(right)

    def compute_indexes(self, length):
        # compute frequency of word,and left/right entropy
        self.tf = self.freq
        self.freq /= length
        self.length = length
        self.left = compute_entropy(self.left)
        self.right = compute_entropy(self.right)

    def compute_pmi(self, words_dict):
        # compute all kinds of combines for word
        sub_part = gen_bigram(self.text)
        # print(sub_part)
        if len(sub_part) > 0:
            # self.pmi = min(
            #     map(lambda x: math.log(self.freq / (words_dict[x[0]].freq *
            #                                         words_dict[x[1]].freq)),
            #         sub_part))
            # print(self.freq, max(words_dict[sub_part[0][0]].freq - self.freq, 0.00001),
            #       max(words_dict[sub_part[0][1]].freq - self.freq, 0.00001))
            # print(math.log(self.freq / (max(words_dict[sub_part[0][0]].freq - self.freq, 0.00001)) *
            #                                    (max(words_dict[sub_part[0][1]].freq - self.freq, 0.00001))))
            self.pmi = min(
                map(
                    lambda x: math.log(self.freq / ((max(words_dict[x[0]].freq - self.freq, 0.00001)) *
                                               (max(words_dict[x[1]].freq - self.freq, 0.00001)))), sub_part
                    )
            )
            # print(self.pmi)
            # os._exit(0)


class segdocument(object):
    '''
    Main class for Chinese word segmentation
    1. Generate words from a long enough document
    2. Do the segmentation work with the document
    reference:

    '''

    def __init__(self, doc, max_word_len=5, min_tf=0.000005, min_entropy=0.07, min_pmi=6):
        super(segdocument, self).__init__()
        self.max_word_len = max_word_len
        self.min_tf = min_tf
        self.min_entropy = min_entropy
        self.min_pmi = min_pmi
        # analysis documents
        self.word_info = self.gen_words(doc)
        count = float(len(self.word_info))
        self.avg_frq = sum(map(lambda w: w.freq, self.word_info)) / count
        self.avg_entropy = sum(map(lambda w: min(w.left, w.right), self.word_info)) / count
        self.avg_pmi = sum(map(lambda w: w.pmi, self.word_info)) / count
        filter_function = lambda f: len(f.text) > 1 and f.tf > self.min_tf and f.pmi > self.min_pmi  \
                                    and min(f.left, f.right) > self.min_entropy
        self.word_tf_pmi_ent = map(lambda w: (w.text, len(w.text), w.tf, w.pmi, min(w.left, w.right)),
                                   filter(filter_function, self.word_info))

    def gen_words(self, doc):
        # pattern = re.compile('[：“。”，！？、《》……；’‘\n——\r\t）、（——^[1-9]d*$]')
        # pattern = re.compile('[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？?：、~@#”“￥：%……&*（）]+|[[A-Za-z0-9]*$]'.decode('utf-8'))
        pattern = re.compile("[^\u4e00-\u9fa5^.^,^，^a-z^A-Z^0-9]")
        pattern = re.compile("[^\u4e00-\u9fa5^.^,^]")
        doc = pattern.sub('', doc)
        doc = list(jieba.cut(doc))
        # doc = pkuseg.pkuseg().cut(doc)
        word_index = extract_cadicateword(doc, self.max_word_len)
        word_cad = {}  # 候选词的字典
        for suffix in tqdm(word_index):
            word = tuple(doc[suffix[0]:suffix[1]])
            if word not in word_cad:
                word_cad[word] = wordinfo(word)
                # record frequency of word and left neighbors and right neighbors
            word_cad[word].update_data(tuple(doc[suffix[0] - 1:suffix[0]]),
                                       tuple(doc[suffix[1]:suffix[1] + 1]))
        length = len(doc)
        # computing frequency of candicate word and entropy of left/right neighbors
        for word in tqdm(word_cad):
            word_cad[word].compute_indexes(length)
        # ranking by length of word

        values = sorted(word_cad.values(), key=lambda x: len(x.text))
        for v in values:
            if len(v.text) == 1:
                continue
            v.compute_pmi(word_cad)
        # ranking by freq
        return sorted(values, key=lambda v: len(v.text), reverse=False)


if __name__ == '__main__':
    starttime = time.clock()
    path = os.path.abspath('.')
    wordlist = []
    word_candidate = []
    dict_bank = []
    dict_path = r'C:\Users\king\Documents\code\NLP\text_classification\sentiment'
    import pickle

    decode_list = ['gb18030', 'gbk', 'utf-8', 'ISO-8859-2', 'unicode']  # providing multiple decode ways
    for decode_way in decode_list:
        try:
            # doc = open(path + '/source.txt', 'r', encoding=decode_way).read()
            doc = open(r'C:\Users\king\Documents\code\data\crawler\result_expand.txt', 'r', encoding=decode_way).read()
            print('Great!{0} success to decode the document!!!'.format(decode_way))
            break
        except:
            print('Oops!{0} cannot decode the document!'.format(decode_way))
    word = segdocument(doc, max_word_len=2, min_tf=2, min_entropy=0.7, min_pmi=6)
    print('平均频率:' + str(word.avg_frq))
    print('平均pmi:' + str(word.avg_pmi))
    print('平均自由度:' + str(word.avg_entropy))

    # for i in open(dict_path, 'r'):
    #     dict_bank.append(i.split(' ')[0])

    print('result:')
    import jieba

    for i in word.word_tf_pmi_ent:
        content = ''.join(list(i[0]))
        if len(content) != len('/'.join(jieba.cut(content))):
            word_candidate.append(content)
            wordlist.append([content, i[1], i[2], i[3], i[4]])
    seg = pd.DataFrame(wordlist, columns=['word', 'length', 'freq', 'pmi', 'entropy'])
    seg.to_excel(path + '/extractword_full.xlsx')
    # intersection = set(word_candidate) & set(dict_bank)
    # newwordset = set(word_candidate) - intersection
    for i in wordlist:
        print(i[0], i[1], i[2], i[3], i[4])

    endtime = time.clock()
    print(endtime - starttime)
