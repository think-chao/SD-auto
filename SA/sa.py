import jieba
import pandas as pd
import os

"""
1. 获取情感词典：basic+extend
2. 否定词词典
3. 程度副词词典
4. 连词词典
"""

import pickle

basic = pickle.load(open('./data/ntsu_hownet.txt', 'rb'))
basic.pop('没有')
basic.pop('不是')
basic.pop('说')
# print(basic['说'])
print('basic dictionary have {} words'.format(len(basic)))

negation = pickle.load(open('./data/negation.txt', 'rb'))
intensity = pickle.load(open('./data/intensity.txt', 'rb'))
conjunction = {'但':-1, '但是':-1, '可':-1, '可是':-1, '然而':-1, '而且':1.5, '并且':1.5, '甚至':1.5, '更加':1.5}



def get_my_senti():
	my_senti = pd.read_excel(r'./data/my_senti.xlsx')
	words, senti = my_senti['word'], my_senti['senti']

	overlap, total = 0, 0
	for i, word in enumerate(words):
		if senti[i] == '[2]':
			senti[i] = -1
		if senti[i] == '[1]':
			senti[i] = 1
		if senti[i] == '[0]':
			senti[i] = 0
		if word in basic:
			if basic[word] != senti[i]:
				basic[word] = senti[i]
			overlap += 1
		else:
			basic[word] = senti[i]
		total += 1
	return basic

def senti_analysis(text, senti_dict):
	
	sentences = text.split(',') if ',' in text else text.split('，')
	total_socre = 0
	for i, sentence in enumerate(sentences):
		# print(sentence)
		sentence_score = 0
		if len(sentence) > 0:
			sentence_split = list(jieba.cut(sentence))
			for i, word in enumerate(sentence_split):
				if word in senti_dict:
					sw = senti_dict[word]
					# print('found senti word {}, {}'.format(word, sw))
					neg, inte = 1, 1
					max_interval_for_negaton, max_interval_for_inten = 0, 0
					for k in range(i-1, -1, -1):
						if sentence_split[k] in negation and max_interval_for_negaton < 2:
							# print('found negation {}'.format(sentence_split[k]))
							neg *= -1
							max_interval_for_negaton = 0
						if sentence_split[k] in intensity and max_interval_for_inten < 2:
							inte *= intensity[sentence_split[k]]
							max_interval_for_inten = 0
						# max_interval_for_negaton += 1
					sentence_score += sw*neg*inte
		total_socre += sentence_score
	if total_socre == 0:
		polarity = '中性'
	elif total_socre > 0:
		polarity = '正向'
	else:
		polarity = '负向'
	print(polarity)
	return polarity

if __name__ == '__main__':
	my_senti_dict = get_my_senti()

	# df_to_analysis = pd.read_excel('./excel_output.xls')
	# df_to_save = {'time':[], 'text':[], 'label':[]}
	# for i, text in enumerate(df_to_analysis['text']):
	# 	df_to_save['time'].append(df_to_analysis['post_time'][i])
	# 	df_to_save['text'].append(df_to_analysis['text'][i])
	# 	df_to_save['label'].append(senti_analysis(text, my_senti_dict))
	# pd.DataFrame(df_to_save).to_excel('label.xlsx')
	senti_analysis('我怎么也没想到我挂科了', my_senti_dict)

