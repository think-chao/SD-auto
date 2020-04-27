from gensim.models import Word2Vec
from gensim.models import word2vec
import re
from tqdm import tqdm
import pkuseg
import pickle

seg = pkuseg.pkuseg()
import jieba
import pandas as pd

cop = re.compile("[^\u4e00-\u9fa5^.^,^，]")

save_path = './data/seg_corpus_new.txt'

# 1.加载语料，去除表情符号，分词
import os


def gen_train_corpus():
    jieba.load_userdict(open('./data/new_words.txt', 'r', encoding='utf-8'))
    if not os.path.exists(save_path):
        f = open(save_path, 'w')
        df_bbs = pd.read_excel('./bbs_corpus.xlsx', sheet_name='content')
        for line in tqdm(df_bbs['内容']):
            if type(line) == str and len(line) >= 3:
                filter_content = cop.sub('', line)
                seg_list = jieba.cut(filter_content)
                f.write(' '.join(seg_list) + '\n')
        f.close()

    topics = ['职业发展', '恋爱关系',
              '学业方面', '师生关系']
    source_excel = './douban_corpus.xlsx'
    source_excel = r'C:\Users\king\Documents\code\NLP\result\result3.xlsx'

    if os.path.exists(save_path):
        f = open(save_path, 'a+')
        for t in topics:
            try:
                df = pd.read_excel(source_excel, sheet_name=t)
            except:
                continue
            sheet = df['content']
            for line in tqdm(sheet):
                if type(line) == str and len(line) >= 3:
                    filter_content = cop.sub('', line)
                    seg_list = jieba.cut(filter_content)
                    f.write(' '.join(seg_list) + '\n')
        f.close()


def train():
    from gensim.models.word2vec import LineSentence
    # gen_train_corpus()

    train_file = './data/seg_corpus_new.txt'
    sentences = LineSentence(open(train_file, 'r'))
    # with open(train_file, 'r') as f:
    #     for line in tqdm(f.readlines()):
    #         seg_list.append(line.strip().split(' '))
    model = Word2Vec(sentences, min_count=3, size=100)
    import pickle
    with open('./data/vec_new.pkl', 'wb') as fp:
        pickle.dump(model, fp)
    # print(model)
    # model.save(model_name)


def test():
    import pickle
    with open('./data/vec_new.pkl', 'rb') as fp:
        vec_model = pickle.load(fp)
    most_similar = vec_model.most_similar(['郁闷'])
    for i, item in enumerate(most_similar):
        print(item)

def vis():
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from matplotlib import pyplot as plt 
    pca=PCA(n_components=100)
    embedded_path = r'C:\Users\king\Documents\code\data\word2vec\27_embeddings.pkl'
    vocab_path = r'C:\Users\king\Documents\code\data\word2vec\27_vocab.pkl'
    with open(embedded_path, 'rb') as fp:
        embeded = pickle.load(fp)
    pca.fit(embeded)

    x = [30, 40, 50, 60, 70, 80, 90, 100]
    y = []
    for v in x:
        print(v, sum(pca.explained_variance_ratio_[:v]))
    # plt.plot(x,y)
    # plt.show()

    # os._exit(0)

    # kmeans = KMeans(n_clusters=3, random_state=0).fit(embeded)
    # print(kmeans.labels_[205:285])


    # with open(vocab_path, 'rb') as fp:
    #     vocab = pickle.load(fp)
    # print(vocab)
    # print(kmeans.predict([embeded[vocab['悲惨']]]))



def main():
    # train()
    test()
    # vis()



if __name__ == '__main__':
    main()
