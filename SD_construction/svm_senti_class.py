import itertools
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import pandas as pd
import numpy as np
# from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

import jieba.analyse
import jieba.posseg as peg
jieba.load_userdict('./new_words.txt')  



def load_new_words_to_jieba():
    new_words_path = r'C:\Users\king\Documents\code\Chinese_segment_augment\New-Word-Detection\extractword1.xlsx'
    df = pd.read_excel(new_words_path)['word']
    save_file = open('./new_words.txt', 'w', encoding='utf-8')
    for word in df:
        save_file.write(word+' '+str(50)+' a\n')



def dic_confusion():
    # 1. 情感词汇本体
    BT = r'C:\Users\king\Desktop\情感词汇本体\情感词汇本体\情感词汇本体.xlsx'
    df = pd.read_excel(BT)
    BT_senti, BT_polarity = df['词语'], df['极性']
    BT_dict = {}
    for i in range(len(BT_senti)):
        polarity = BT_polarity[i]
        if polarity in [0, 1, 2]:
            BT_dict[BT_senti[i]] = polarity

    # 2. 知网和NTSUD
    hownet_ntsud = pickle.load(
        open(r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\ntsu_hownet.txt', 'rb')
    )

    fused_dict = {}
    for word in BT_dict:
        if hownet_ntsud.get(word, 0) in [-1, 1]:
            hownet_ntsud_p = 2 if hownet_ntsud[word] == -1 else 1
            if hownet_ntsud_p == BT_dict[word]:
                fused_dict[word] = hownet_ntsud_p
            else:
                print(word)
        else:
            fused_dict[word] = BT_dict[word]
    os._exit(0)
    print('基础词典中词的总量：{} 其中'.format(len(fused_dict)))
    print('褒义词数目')
    print(sum([1 if fused_dict[v] == 1 else 0 for v in fused_dict]))
    print('贬义词数目')
    print(sum([1 if fused_dict[v] == 2 else 0 for v in fused_dict]))
    print('中性词数目')
    print(sum([1 if fused_dict[v] == 0 else 0 for v in fused_dict]))
    return fused_dict


def gen_stop_words():
    source = r'C:\Users\king\Documents\code\NLP\text_classification\stopwords\百度停用词表.txt'
    stop_dict = {}
    for line in open(source, 'r', encoding='utf-8').readlines():
        stop_dict[line.strip()] = stop_dict.get(line.strip(), 0) + 1
    return stop_dict


def gen_corpus(basic, stop, neg, inten):
    import random
    corpus_dic = pickle.load(
        open(r'C:\Users\king\Documents\code\data\word2vec\27_vocab.pkl', 'rb'))
    corpus_embedding = pickle.load(
        open(r'C:\Users\king\Documents\code\data\word2vec\27_embeddings.pkl', 'rb'))

    train_data = []
    train_label = []
    test_data = []
    for word in corpus_dic:
        if word in stop or word in neg or word in inten:
            continue
        if word in basic:
            # print(word, basic[word])
            # if basic[word] != 0:
            if basic[word] == 0:
                print(word)
            train_data.append(corpus_embedding[corpus_dic[word]])
            train_label.append(basic[word])
        else:
            test_data.append((word, corpus_embedding[corpus_dic[word]]))

    # train_data.append(corpus_embedding[corpus_dic['惨']])
    # train_label.append(1)
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    print(len(train_label[train_label == 0]))
    print(len(train_label[train_label == 1]))
    print(len(train_label[train_label == 2]))
    os._exit(0)


    clf = SVC(gamma='auto', probability=True, kernel='rbf')
    # clf = GaussianNB()
    # clf = tree.DecisionTreeClassifier()
    # clf = RandomForestClassifier(oob_score=True, random_state=10)
    clf.fit(train_data, train_label)
    prev_score = 0
    while True:
        cur_score = clf.score(train_data, train_label)
        print(cur_score)
        if cur_score < prev_score:
            break
        # random.shuffle(test_data)
        need_label = {}
        already_label = []
        n = 0
        for i, test_case in enumerate(test_data):
            # print(clf.predict([test_case[1]]), clf.decision_function([test_case[1]]))
            # print(test_case[0], clf.decision_function([test_case[1]]), clf.predict_proba([test_case[1]]))
            # os._exit(0)
            if test_case[0] in stop or test_case[0] in inten or test_case[0] in already_label:
                continue
            words = peg.cut(test_case[0])
            tag = str(list(words)[0]).split('/')[1]
            if len(test_case[0]) == 1 and tag in ['n', 'nz', 'v', 'vd', 'vn']:
                continue
            if tag in ['a', 'ad', 'an','v', 'vd', 'vn']:
                cls_prob = clf.predict_proba([test_case[1]])
                cls_pred = clf.predict([test_case[1]])
                need_label[test_case[0]] = [max(cls_prob[0])-min(cls_prob[0]), test_case[1], cls_pred]
                print(test_case[0], cls_pred)
                n += 1
            # AL Strategy： 取最大概率和最小概率之差最小的前N个样本
        print('*********************************************************')
        print(n)
        print('*********************************************************')

        hardestNSample = sorted(need_label.keys(), key=lambda x: need_label[x][0])[:30]
        for record in hardestNSample:
            already_label.append(need_label[record][0])
            train_data = np.append(train_data, [need_label[record][1]], axis=0)
            train_label = np.append(train_label, int(input('样本: {}, 预测：{}，输入正确标签\n'.format(record, need_label[record][2]))))
        clf.fit(train_data, train_label)



    # os._exit(0)

    from sklearn import metrics
    cm_train = metrics.confusion_matrix(train_label, clf.predict(train_data))
    
    plt.figure()
    plot_confusion_matrix(cm_train, normalize=True)


def plot_confusion_matrix(cm, classes=[0, 1],
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def gen_adv_words():
    import pickle
    negation = pickle.load(
        open(r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\negation.txt', 'rb'))
    intensity = pickle.load(
        open(r'C:\Users\king\Documents\code\NLP\text_classification\sentiment\intensity.txt', 'rb'))
    return negation, intensity
    

if __name__ == '__main__':
    # load_new_words_to_jieba()
    # os._exit(0)
    basic_dict, stop_dict = dic_confusion(), gen_stop_words()
    neg_dict, inten_dict = gen_adv_words()
    gen_corpus(basic_dict, stop_dict, neg_dict, inten_dict)
