import jieba.analyse
import jieba.posseg as peg
jieba.load_userdict('./new_words.txt')  

words = peg.cut('事业编')
tag = str(list(words)).split('/')
print(tag)